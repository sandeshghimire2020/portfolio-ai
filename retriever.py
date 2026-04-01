import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "sandesh_knowledge"
EMBED_MODEL = "text-embedding-3-small"
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.5"))

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def embed(text: str) -> list[float]:
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
    total_docs = collection.count()
    if total_docs == 0:
        return []

    query_embedding = embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_docs),
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])
    if not documents or not metadatas or not distances:
        return []

    chunks = []
    for doc, meta, dist in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        if dist <= RELEVANCE_THRESHOLD:
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", "unknown"),
                "score": round(dist, 4),
            })

    return chunks
