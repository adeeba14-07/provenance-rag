from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re


def tokenize(text):
    """Simple tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def bm25_search(query, documents, metadatas, top_k=10):
    """Keyword-based search using BM25."""
    if not documents:
        return [], []

    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True
    )[:top_k]

    return [r[0] for r in ranked], [r[1] for r in ranked]


def vector_search(query, collection, embedder, top_k=10):
    """Semantic search using ChromaDB."""
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    return results["documents"][0], results["metadatas"][0]


def reciprocal_rank_fusion(vector_docs, vector_metas, bm25_docs, bm25_metas, k=60):
    """Combine rankings from both retrievers using RRF."""
    scores = {}

    for rank, (doc, meta) in enumerate(zip(vector_docs, vector_metas)):
        key = doc[:200]
        if key not in scores:
            scores[key] = {"doc": doc, "meta": meta, "score": 0}
        scores[key]["score"] += 1 / (k + rank + 1)

    for rank, (doc, meta) in enumerate(zip(bm25_docs, bm25_metas)):
        key = doc[:200]
        if key not in scores:
            scores[key] = {"doc": doc, "meta": meta, "score": 0}
        scores[key]["score"] += 1 / (k + rank + 1)

    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["doc"] for r in sorted_results], [r["meta"] for r in sorted_results]


def rerank(query, documents, metadatas, top_k=5):
    """Rerank using a cross-encoder model."""
    if not documents:
        return [], []

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True
    )[:top_k]

    return [r[0] for r in ranked], [r[1] for r in ranked]


def hybrid_retrieve(query, collection, embedder, top_k=5, candidate_k=15):
    """Full pipeline: vector + BM25 + RRF + rerank."""
    # Step 1: Vector search
    vector_docs, vector_metas = vector_search(query, collection, embedder, top_k=candidate_k)

    # Step 2: Get all docs for BM25 from the same collection
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    # Step 3: BM25 search
    if all_docs:
        bm25_docs, bm25_metas = bm25_search(query, all_docs, all_metas, top_k=candidate_k)
    else:
        bm25_docs, bm25_metas = [], []

    # Step 4: Combine with RRF
    combined_docs, combined_metas = reciprocal_rank_fusion(
        vector_docs, vector_metas, bm25_docs, bm25_metas
    )

    # Step 5: Rerank
    final_docs, final_metas = rerank(query, combined_docs, combined_metas, top_k=top_k)

    return final_docs, final_metas