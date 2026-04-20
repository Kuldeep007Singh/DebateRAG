# RAG/hybrid_retriever.py

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from RAG.vectorstore import query_vectorstore
from typing import List, Dict

# Cross-encoder for re-ranking — loads once at import time
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def bm25_search(query: str, all_chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    scored = [(score, chunk) for score, chunk in zip(scores, all_chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # ── normalize to match ChromaDB output format ──
    normalized = []
    for _, chunk in scored[:top_k]:
        if "metadata" not in chunk:
            chunk = {
                "text"     : chunk["text"],
                "score"    : 0.0,
                "metadata" : {
                    "title"    : chunk.get("title", "Unknown"),
                    "authors"  : ", ".join(chunk["authors"]) if isinstance(chunk.get("authors"), list) else chunk.get("authors", "Unknown"),
                    "url"      : chunk.get("url", ""),
                    "paper_id" : chunk.get("paper_id", ""),
                    "source"   : chunk.get("source", "arxiv"),
                }
            }
        normalized.append(chunk)
    
    return normalized


def hybrid_search(
    query: str,
    all_chunks: List[Dict],
    top_k: int = 10,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> List[Dict]:
    """
    Combines dense (ChromaDB) + BM25 results using weighted score fusion.
    """
    # Dense retrieval from ChromaDB
    dense_results = query_vectorstore(query, n_results=top_k)
    
    # BM25 keyword retrieval
    bm25_results = bm25_search(query, all_chunks, top_k=top_k)

    
    # Build score maps keyed by chunk text (used as unique id proxy)
    dense_scores = {}
    for rank, chunk in enumerate(dense_results):
        key = chunk["text"][:100]  # first 100 chars as key
        dense_scores[key] = (1 / (rank + 1)) * dense_weight  # reciprocal rank
    
    bm25_scores = {}
    for rank, chunk in enumerate(bm25_results):
        key = chunk["text"][:100]
        bm25_scores[key] = (1 / (rank + 1)) * bm25_weight
    
    # Merge — union of both result sets
    all_keys = set(dense_scores.keys()) | set(bm25_scores.keys())
    
    # Build unified chunk map
    chunk_map = {}
    for chunk in dense_results + bm25_results:
        key = chunk["text"][:100]
        if key not in chunk_map:
            chunk_map[key] = chunk
    
    # Compute fused scores
    fused = []
    for key in all_keys:
        score = dense_scores.get(key, 0) + bm25_scores.get(key, 0)
        if key in chunk_map:
            fused.append((score, chunk_map[key]))
    
    fused.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in fused[:top_k]]


def rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Cross-encoder re-ranking — scores every (query, chunk) pair directly.
    Much more accurate than bi-encoder similarity alone.
    """
    if not chunks:
        return []
    
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = RERANKER.predict(pairs)
    
    # Attach rerank score to each chunk
    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [chunk for _, chunk in scored[:top_k]]


def retrieve_and_rerank(
    query      : str,
    all_chunks : List[Dict],
    top_k      : int = 5,
    stance     : str = None   # ← new optional param
) -> List[Dict]:
    """
    Full pipeline: query rewriting → hybrid search → cross-encoder rerank.
    """
    from RAG.query_rewriter import rewrite_query

    # Step 1 — rewrite query into 3 optimized variants if stance provided
    if stance:
        queries = rewrite_query(topic=query, stance=stance)
        print(f"[QueryRewriter] Rewritten queries: {queries}")
    else:
        queries = [query]

    # Step 2 — hybrid search for each query, collect all candidates
    all_candidates = []
    seen_texts     = set()

    for q in queries:
        candidates = hybrid_search(q, all_chunks, top_k=top_k * 2)
        for chunk in candidates:
            key = chunk["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_candidates.append(chunk)

    # Step 3 — rerank the merged candidate pool
    reranked = rerank(query, all_candidates, top_k=top_k)
    return reranked