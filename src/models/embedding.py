from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
_model = None

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Load and cache the embedding model so it's not loaded multiple times."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_chunks(chunks: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Embed a list of text chunks. Returns Numpy array of shape (num_chunks, embedding_dim)
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    return embeddings

def embed_query(query: str, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Embed a single query string. Returns embedding vector.
    """
    model = get_embedding_model(model_name)
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0]

def save_embeddings(embeddings: np.ndarray, filename: str):
    """
    Save Numpy embeddings to a compressed .npz file.
    """
    np.savez_compressed(filename, embeddings=embeddings)

def load_embeddings(filename: str) -> np.ndarray:
    """
    Load embeddings from a .npz file.
    """
    data = np.load(filename)
    return data["embeddings"]

