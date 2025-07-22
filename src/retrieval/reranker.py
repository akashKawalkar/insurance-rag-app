from typing import List, Dict
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv()  
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME")
RERANKER_MODEL_NAME = RERANKER_MODEL_NAME

_reranker_model = None

def get_reranker_model(model_name: str = RERANKER_MODEL_NAME):
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model

def rerank(
    query: str,
    candidate_chunks: List[Dict], 
    chunk_field: str = "chunk",
    top_k: int = 5,
    model_name: str = RERANKER_MODEL_NAME
) -> List[Dict]:
    """
    Rerank candidate chunks using a cross-encoder.
    
    Args:
        query: User question.
        candidate_chunks: List of dicts with at least key chunk_field.
        chunk_field: Field name for the text in each candidate dict.
        top_k: Return only top K results.
        model_name: Cross-encoder model name.

    Returns:
        Sorted list of candidate dicts, each with an added 'rerank_score'.
    """
    model = get_reranker_model(model_name)
    pairs = [(query, c[chunk_field]) for c in candidate_chunks]
    scores = model.predict(pairs)
    for c, s in zip(candidate_chunks, scores):
        c["rerank_score"] = float(s)
    sorted_chunks = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_chunks[:top_k]

