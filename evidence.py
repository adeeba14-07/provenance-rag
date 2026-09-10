import re
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def split_into_claims(answer):
    """Break an answer into individual factual claims."""
    # Remove citations and formatting
    cleaned = re.sub(r'【.*?】', '', answer)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    cleaned = re.sub(r'\*\*.*?\*\*', '', cleaned)
    cleaned = re.sub(r'#+ ', '', cleaned)

    # Split by sentences and bullets
    lines = re.split(r'[.\n•\-]', cleaned)
    claims = [line.strip() for line in lines if len(line.strip()) > 20]

    return claims[:10]  # Limit to top 10 claims


def verify_claims(claims, context_text, groq_client):
    """Verify each claim against the retrieved context."""
    if not claims:
        return []

    claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])

    prompt = f"""You are an evidence verification system. Your job is to check whether each claim is supported by the provided context.

CONTEXT (from user's uploaded document):
{context_text[:4000]}

CLAIMS TO VERIFY:
{claims_text}

For EACH claim, respond in this exact format:
CLAIM: [claim text]
STATUS: SUPPORTED / PARTIAL / UNSUPPORTED
REASON: [one sentence explanation]

Rules:
- SUPPORTED = context directly contains the information
- PARTIAL = context contains related info but not exactly
- UNSUPPORTED = context does not contain this information

Be strict. Do not mark anything as SUPPORTED unless the context explicitly contains it.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1500
    )

    raw = response.choices[0].message.content
    return parse_verification_results(raw, claims)


def parse_verification_results(raw_text, original_claims):
    """Parse the LLM verification output into structured data."""
    results = []
    blocks = raw_text.split("CLAIM:")

    for block in blocks[1:]:
        try:
            lines = block.strip().split("\n")
            claim_text = lines[0].strip()

            status = "UNSUPPORTED"
            reason = ""

            for line in lines:
                if line.startswith("STATUS:"):
                    status = line.replace("STATUS:", "").strip().upper()
                    if "SUPPORTED" in status and "UN" not in status and "PARTIAL" not in status:
                        status = "SUPPORTED"
                    elif "PARTIAL" in status:
                        status = "PARTIAL"
                    else:
                        status = "UNSUPPORTED"
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()

            results.append({
                "claim": claim_text,
                "status": status,
                "reason": reason
            })
        except Exception:
            continue

    return results


def compute_trust_score(verification_results):
    """Compute trust score from verification results."""
    if not verification_results:
        return 0, {"supported": 0, "partial": 0, "unsupported": 0, "total": 0}

    supported = sum(1 for r in verification_results if r["status"] == "SUPPORTED")
    partial = sum(1 for r in verification_results if r["status"] == "PARTIAL")
    unsupported = sum(1 for r in verification_results if r["status"] == "UNSUPPORTED")
    total = len(verification_results)

    score = int(((supported * 100) + (partial * 50)) / total) if total > 0 else 0

    return score, {
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "total": total
    }


def run_evidence_verification(answer, context_text, groq_api_key):
    """Full pipeline: split, verify, score."""
    groq_client = Groq(api_key=groq_api_key)

    claims = split_into_claims(answer)
    if not claims:
        return {
            "claims": [],
            "trust_score": 0,
            "summary": {"supported": 0, "partial": 0, "unsupported": 0, "total": 0}
        }

    verification_results = verify_claims(claims, context_text, groq_client)
    trust_score, summary = compute_trust_score(verification_results)

    return {
        "claims": verification_results,
        "trust_score": trust_score,
        "summary": summary
    }