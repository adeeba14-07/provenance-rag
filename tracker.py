import json
import os
from datetime import datetime

TRACKER_FILE = "provenance_stats.json"


def load_stats():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {
        "documents_uploaded": 0,
        "total_chunks": 0,
        "total_queries": 0,
        "queries_history": [],
        "documents_history": [],
        "total_retrieval_time_ms": 0,
        "total_generation_time_ms": 0,
        "verification_results": {"verified": 0, "partial": 0, "flagged": 0},
        "trust_scores": [],
        "last_updated": None
    }


def save_stats(stats):
    stats["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRACKER_FILE, "w") as f:
        json.dump(stats, f, indent=4)


def log_document_upload(filename, chunks_count, source_type="file"):
    stats = load_stats()
    stats["documents_uploaded"] += 1
    stats["total_chunks"] += chunks_count
    stats["documents_history"].append({
        "filename": filename,
        "chunks": chunks_count,
        "source_type": source_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_stats(stats)


def log_query(question, answer, sources, retrieval_ms, generation_ms,
              verification="none", trust_score=0, raw_chunks=None):
    stats = load_stats()
    stats["total_queries"] += 1
    stats["total_retrieval_time_ms"] += retrieval_ms
    stats["total_generation_time_ms"] += generation_ms

    if verification == "verified":
        stats["verification_results"]["verified"] += 1
    elif verification == "partial":
        stats["verification_results"]["partial"] += 1
    elif verification == "flagged":
        stats["verification_results"]["flagged"] += 1

    stats.setdefault("trust_scores", []).append(trust_score)

    stats["queries_history"].append({
        "question": question,
        "answer": answer[:800],
        "sources": sources,
        "raw_chunks": raw_chunks if raw_chunks else [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "verification": verification,
        "trust_score": trust_score
    })
    save_stats(stats)


def get_average_retrieval_time():
    stats = load_stats()
    if stats["total_queries"] > 0:
        return round(stats["total_retrieval_time_ms"] / stats["total_queries"], 2)
    return 0


def get_average_generation_time():
    stats = load_stats()
    if stats["total_queries"] > 0:
        return round(stats["total_generation_time_ms"] / stats["total_queries"], 2)
    return 0


def get_average_trust_score():
    stats = load_stats()
    scores = stats.get("trust_scores", [])
    if scores:
        return int(sum(scores) / len(scores))
    return 0