import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_embedder_cache = None

def get_embedder():
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder_cache


def compute_faithfulness(answer, context_text, groq_api_key):
    """Reuse the deterministic verifier's score for faithfulness."""
    try:
        from verifier import verify_answer, get_embedder
        # Split context_text into chunks for the verifier
        chunks = [c.strip() for c in context_text.split("\n\n") if len(c.strip()) > 50]
        if not chunks:
            chunks = [context_text]

        metadatas = [{"source": "evaluation", "location": "Section"} for _ in chunks]
        result = verify_answer(answer, chunks, metadatas, groq_api_key, "")
        return result["trust_score"] / 100.0
    except Exception:
        return 0.0

def compute_answer_relevancy(question, answer):
    """Cosine similarity between question and answer embeddings."""
    try:
        embedder = get_embedder()
        emb_q = embedder.encode([question])
        emb_a = embedder.encode([answer])
        score = cosine_similarity(emb_q, emb_a)[0][0]
        return round(float(score), 3)
    except Exception:
        return 0.0


def compute_context_precision(question, contexts):
    """How many retrieved chunks are actually relevant to the question."""
    if not contexts:
        return 0.0
    try:
        embedder = get_embedder()
        emb_q = embedder.encode([question])
        emb_c = embedder.encode(contexts)
        sims = cosine_similarity(emb_q, emb_c)[0]
        relevant = sum(1 for s in sims if s > 0.3)
        return round(relevant / len(contexts), 3)
    except Exception:
        return 0.0


def compute_context_recall(answer, contexts):
    """How much of the answer is grounded in the retrieved contexts."""
    if not contexts:
        return 0.0
    try:
        embedder = get_embedder()
        answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
        if not answer_sentences:
            return 0.0

        emb_a = embedder.encode(answer_sentences)
        emb_c = embedder.encode(contexts)

        grounded_count = 0
        for i in range(len(answer_sentences)):
            sims = cosine_similarity([emb_a[i]], emb_c)[0]
            if max(sims) > 0.5:
                grounded_count += 1

        return round(grounded_count / len(answer_sentences), 3)
    except Exception:
        return 0.0


def run_custom_evaluation(test_cases, groq_api_key):
    """Run custom evaluation on multiple test cases."""
    if not test_cases:
        return None

    results = []
    for tc in test_cases:
        question = tc.get("question", "")
        answer = tc.get("answer", "")
        contexts = tc.get("contexts", [])

        context_text = "\n".join(contexts)

        faithfulness = compute_faithfulness(answer, context_text, groq_api_key)
        answer_relevancy = compute_answer_relevancy(question, answer)
        context_precision = compute_context_precision(question, contexts)
        context_recall = compute_context_recall(answer, contexts)

        results.append({
            "question": question,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        })

    # Average across all test cases
    n = len(results)
    if n == 0:
        return None

    return {
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 3),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 3),
        "context_precision": round(sum(r["context_precision"] for r in results) / n, 3),
        "context_recall": round(sum(r["context_recall"] for r in results) / n, 3),
        "total_cases": n,
        "per_case": results,
    }