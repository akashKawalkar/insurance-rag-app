import json
from typing import List, Dict
from rapidfuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv() 

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
CHUNK_JSON_PATH = os.environ.get("CHUNK_JSON_PATH")
# ---- CONFIG ----
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME 
TAG_MATCH_THRESHOLD = 80   
TAG_SCORE_BOOST = 0.15   
TOP_N_RESULTS = 5

def parse_tags(tag_field: str) -> List[str]:
    """
    Turn raw tag string (e.g., "- tag1\n- tag2\n") into clean list.
    """
    return [line.lstrip("-").strip() for line in tag_field.strip().split('\n') if line.strip()]

def tags_match(chunk_tags: List[str], query_tags: List[str], threshold: int = TAG_MATCH_THRESHOLD) -> bool:
    """
    Returns True if any tag matches the query tags above the threshold.
    """
    for ctag in chunk_tags:
        ctag_stripped = ctag.lower()
        for qtag in query_tags:
            qtag_stripped = qtag.lower()
            score = fuzz.token_set_ratio(ctag_stripped, qtag_stripped)
            if score >= threshold:
                return True
    return False

def retrieve_chunks(
    jsonl_path: str,
    query: str,
    query_tags: List[str],
    top_n: int = TOP_N_RESULTS,
    model_name: str = EMBEDDING_MODEL_NAME
) -> List[Dict]:
    """
    Hybrid retrieval of chunks using both embedding and tag match.
    Returns top-N scored chunks.
    """
    model = SentenceTransformer(model_name)

    # 1. Load chunks + tags
    chunks, tags_per_chunk = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj["chunk"])
            tags_per_chunk.append(parse_tags(obj["tags"]))

    # 2. Embed all chunks and the query
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    # 3. Compute embedding similarity (cosine)
    emb_scores = np.dot(chunk_embeddings, query_emb)

    # 4. Tag match for each chunk
    tag_matches = [tags_match(tags, query_tags) for tags in tags_per_chunk]

    # 5. Score & collect results
    results = []
    for idx, (score, tag_match) in enumerate(zip(emb_scores, tag_matches)):
        final_score = score + (TAG_SCORE_BOOST if tag_match else 0)
        source = (
            "BOTH" if tag_match and score > 0 else
            "TAG" if tag_match else
            "EMBEDDING"
        )
        results.append({
            "score": final_score,
            "source": source,
            "chunk": chunks[idx],
            "tags": tags_per_chunk[idx]
        })

    # 6. Top-N by score, descending
    best = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    return best

