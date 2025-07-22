import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv() 

QDRANT_CLOUD_URL = os.environ.get("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
COLLECTION = os.environ.get("QDRANT_COLLECTION")
CHUNK_JSON_PATH = os.environ.get("CHUNK_JSON_PATH")

#CONFIG
COLLECTION = COLLECTION
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
INPUT_PATH = CHUNK_JSON_PATH


CLOUD_URL = QDRANT_CLOUD_URL
API_KEY = QDRANT_API_KEY

client = QdrantClient(
    url=CLOUD_URL,
    api_key=API_KEY
)

def main():
    # 1. Load data
    chunks, metas = [], []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj["chunk"])
            metas.append({key: obj[key] for key in ("tags", "rhde_queries", "chunk")})

    # 2. Embed chunks
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)

    # 3. (Re)create Qdrant collection in the cloud
    client = QdrantClient(
        url=CLOUD_URL,
        api_key=API_KEY
    )
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )

    # 4. Upload in points:
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload=metas[i]
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Uploaded {len(points)} chunks with vectors and metadata.")

if __name__ == "__main__":
    main()
