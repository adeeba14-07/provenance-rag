import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean, median

_embedder = None


def get_embedder():
    """Load the embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ============================================
# CALCULATION HANDLING
# ============================================
def detect_calculation(query):
    q = query.lower()
    if any(w in q for w in ["average", "mean", "avg"]):
        return "average"
    if any(w in q for w in ["sum", "total", "add up"]):
        return "sum"
    if any(w in q for w in ["maximum", "max", "highest", "largest"]):
        return "max"
    if any(w in q for w in ["minimum", "min", "lowest", "smallest"]):
        return "min"
    if any(w in q for w in ["count", "how many", "number of"]):
        return "count"
    if any(w in q for w in ["median"]):
        return "median"
    return None


def extract_numbers(chunks):
    numbers = []
    for chunk in chunks:
        for match in re.findall(r'(?:Yes|No)\s*[:=]\s*([\d.]+)\s*%?', chunk):
            try:
                numbers.append(float(match))
            except ValueError:
                continue
    return numbers


def compute_calc(intent, numbers):
    if not numbers:
        return None, "No numeric values found."
    try:
        if intent == "average":
            return round(mean(numbers), 2), f"Average of {len(numbers)} values"
        if intent == "sum":
            return round(sum(numbers), 2), f"Sum of {len(numbers)} values"
        if intent == "max":
            return max(numbers), f"Maximum of {len(numbers)} values"
        if intent == "min":
            return min(numbers), f"Minimum of {len(numbers)} values"
        if intent == "count":
            return len(numbers), f"Count of {len(numbers)} values"
        if intent == "median":
            return round(median(numbers), 2), f"Median of {len(numbers)} values"
    except Exception as e:
        return None, f"Calculation error: {e}"
    return None, "Unknown intent."


# ============================================
# CLAIM EXTRACTION (deterministic)
# ============================================
SKIP_PHRASES = [
    "let me know", "here's what", "if you're looking", "if you need",
    "feel free", "i'd be happy", "hope this", "does that help",
    "in a nutshell", "sources used", "based on the excerpts",
    "according to the excerpts", "the document does not contain",
    "does not contain information", "cannot answer",
    "i don't have enough information", "the provided document does not",
    "in short", "bottom line", "key observations", "in summary",
    "take-aways", "takeaways", "overall", "the following",
    "differences between", "comparison of", "overview of", "summary of",
    "these are the", "this is the", "all information above",
    "as described in", "as stated in" , "datasets mentioned", "these are all", "these are the",
    "in summary", "key differences", "as described in"
]


def clean_markdown(text):
    if not text:
        return ""
    # Strip LaTeX math (which never matches source text)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^\$]+\$', '', text)
    text = re.sub(r'\\frac\{[^}]*\}\{[^}]*\}', '', text)
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\sum[^\s]*', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[Source:[^\]]*\]', '', text)
    text = re.sub(r'\[Chunk:[^\]]*\]', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    for _ in range(3):
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
    text = text.replace('**', '').replace('*', '').replace('__', '')
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
                # Catch separator rows
        stripped_dashes = s.replace("|", "").replace(" ", "").strip()
        if stripped_dashes and set(stripped_dashes) <= set("-:"):
            continue
        if s.startswith("|"):
            # Detect separator rows first: |---|---| or |---|:---|
            raw_inner = s.strip("|").strip()
            if set(raw_inner.replace(" ", "").replace("|", "")) <= set("-:"):
                continue  # this is a separator row

            cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
            if not cells:
                continue

            # Header detection: all cells are short labels, no numbers, no verbs
            all_short = all(len(c) < 30 for c in cells)
            no_numbers = not any(any(ch.isdigit() for ch in c) for c in cells)
            no_verbs = not any(
                any(v in c.lower().split() for v in [
                    "is", "are", "was", "were", "has", "have",
                    "uses", "used", "creates", "created",
                    "shows", "show", "captures", "capture",
                    "supports", "support", "trained", "trains",
                    "resampled", "derived", "obtained", "selected",
                    "collected", "recorded", "measured", "evaluated"
                ])
                for c in cells
            )
            # Also treat short labels like "Period", "Sampling", "Interval", "Source" as header markers
            header_words = {"period", "sampling", "interval", "source", "dataset",
                            "aspect", "purpose", "origin", "description", "how",
                            "used", "value", "year", "state", "yes", "no",
                            "category", "condition", "location", "row", "type"}
            all_header_words = all(
                any(w in c.lower() for w in header_words)
                for c in cells if c.split()
            )

            if (all_short and no_numbers and no_verbs) or all_header_words:
                continue  # skip header row

            s = " ".join(cells)
        s = re.sub(r'^[-•*]\s+', '', s)
        s = re.sub(r'^\d+\.\s+', '', s)
        s = s.strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def extract_claims(answer):
    """Extract clean factual claims from the AI's answer."""
    if not answer or len(answer.strip()) < 20:
        return []
    cleaned = clean_markdown(answer)
    claims = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if len(line) < 20:
            continue
        if any(line.lower().startswith(p) for p in SKIP_PHRASES):
            continue

        # NEW: if this line contains pipes, it is a table row. Split cells.
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            for cell in cells:
                if len(cell) < 25:
                    continue
                if any(p in cell.lower() for p in SKIP_PHRASES):
                    continue
                # Also split cell on sentence boundaries
                for sub in re.split(r'(?<=[.!?])\s+(?=[A-Z])', cell):
                    sub = sub.strip().strip('.').strip()
                    if len(sub) >= 25:
                        claims.append(sub)
            continue

        # Normal lines: split on sentence + separator boundaries
        for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line):
            s = s.strip()
            if len(s) < 25:
                continue
            if any(p in s.lower() for p in SKIP_PHRASES):
                continue
            parts = re.split(r'\s*[/•—]\s*|\s+and\s+', s)
            for p in parts:
                p = p.strip().strip('.').strip()
                p = re.sub(r'^[-•*]\s*', '', p).strip()
                if len(p) >= 20:
                    claims.append(p)
    seen = set()
    unique = []
    for c in claims:
        key = c[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:12]


# ============================================
# COSINE-SIMILARITY VERIFICATION (no LLM)
# ============================================
def verify_claims_deterministic(claims, chunks, metadatas):
    """Verify claims using cosine similarity between claim and chunks."""
    if not claims or not chunks:
        return []

    embedder = get_embedder()
    chunk_embeddings = embedder.encode(chunks)
    claim_embeddings = embedder.encode(claims)

        # Precompute all chunk text as one big string for keyword fallback
    all_context = " ".join(chunks).lower()

    results = []
    for i, claim in enumerate(claims):
        sims = cosine_similarity([claim_embeddings[i]], chunk_embeddings)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        # Keyword fallback for list-style claims (short, many numbers/dates)
        if best_score < 0.55:
            tokens = [t for t in re.findall(r'[A-Za-z]{4,}|\d{4}', claim) if len(t) >= 4]
            if tokens:
                hits = sum(1 for t in tokens if t.lower() in all_context)
                keyword_ratio = hits / len(tokens)
                # If 70%+ of key tokens appear in the source, boost the score
                if keyword_ratio >= 0.70:
                    best_score = max(best_score, 0.60)
                elif keyword_ratio >= 0.50:
                    best_score = max(best_score, 0.45)

        if best_score >= 0.55:
            status = "SUPPORTED"
            conf = "HIGH" if best_score >= 0.70 else "MEDIUM"
        elif best_score >= 0.40:
            status = "PARTIAL"
            conf = "MEDIUM" if best_score >= 0.50 else "LOW"
        else:
            status = "UNSUPPORTED"
            conf = "LOW"

        meta = metadatas[best_idx]
        citation = {
            "source": meta.get("source", "unknown"),
            "location": meta.get("location", "Section"),
            "chunk": best_idx + 1,
            "confidence": conf
        }

        # Detect calculation claims
        ctype = "FACTUAL"
        if re.search(r'\d+\.?\d*\s*[-+*/=]\s*\d+', claim) or \
           any(w in claim.lower() for w in ["average", "total", "sum", "mean"]):
            ctype = "CALCULATION"

        results.append({
            "claim": claim,
            "type": ctype,
            "status": status,
            "reason": f"Similarity {best_score:.2f} to source chunk {best_idx + 1}",
            "citation": citation,
            "similarity": round(best_score, 3)
        })

    return results


# ============================================
# SCORING
# ============================================
def compute_score(results):
    factual = [r for r in results if r.get("type") != "INTERPRETATION"]
    if not factual:
        return 100, {"supported": 0, "partial": 0, "unsupported": 0,
                     "total": 0, "interpretations": len(results),
                     "note": "No verifiable claims extracted"}
    s = sum(1 for r in factual if r["status"] == "SUPPORTED")
    p = sum(1 for r in factual if r["status"] == "PARTIAL")
    u = sum(1 for r in factual if r["status"] == "UNSUPPORTED")
    total = len(factual)
    # Lenient scoring: supported = 100, partial = 75, unsupported = 20
    raw = (s * 100 + p * 75 + u * 20) / total
    score = int(min(raw, 100))
    return score, {
        "supported": s, "partial": p, "unsupported": u,
        "total": total, "interpretations": len(results) - len(factual)
    }


# ============================================
# MAIN ENTRY POINT
# ============================================
def verify_answer(answer, chunks, metas, groq_api_key=None, query=""):
    claims = extract_claims(answer)
    calc_claim = None

    intent = detect_calculation(query)
    if intent:
        nums = extract_numbers(chunks)
        result, explanation = compute_calc(intent, nums)
        if result is not None:
            calc_claim = {
                "claim": f"{intent.title()}: {result} ({explanation})",
                "type": "CALCULATION",
                "status": "SUPPORTED",
                "reason": f"Computed in Python from {len(nums)} values in source.",
                "citation": None,
                "similarity": 1.0
            }

    if not claims and not calc_claim:
        q = query.lower()
        is_summary = any(w in q for w in ["summar", "what is this", "overview",
                                          "tell me about", "datasets", "difference"])
        note = ("Summary question — no specific factual claims to verify."
                if is_summary else "No verifiable claims extracted.")
        return {"claims": [], "trust_score": 100,
                "summary": {"supported": 0, "partial": 0, "unsupported": 0,
                            "total": 0, "interpretations": 0, "note": note}}

    if claims:
        results = verify_claims_deterministic(claims, chunks, metas)
    else:
        results = []

    if calc_claim:
        results.insert(0, calc_claim)

    score, summary = compute_score(results)
    return {"claims": results, "trust_score": score, "summary": summary}