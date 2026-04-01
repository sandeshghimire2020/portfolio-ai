import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "sandesh_knowledge"


def inspect_collection() -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])

    total = len(all_data["ids"])
    print(f"\n{'=' * 60}")
    print(f"  ChromaDB at: {CHROMA_PATH}")
    print(f"  Collection:  {COLLECTION_NAME}")
    print(f"  Total chunks stored: {total}")
    print(f"{'=' * 60}\n")

    for i, (doc_id, doc, meta, embedding) in enumerate(
        zip(
            all_data["ids"],
            all_data["documents"],
            all_data["metadatas"],
            all_data["embeddings"],
        ),
        start=1,
    ):
        print(f"-- Chunk {i} of {total} {'-' * 40}")
        print(f"  ID:        {doc_id}")
        print(f"  Source:    {meta.get('source')}")
        print(f"  Section:   {meta.get('section')}")
        print(f"  Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, ... ] ({len(embedding)} dims)")
        print("  Text preview:")
        preview = doc[:300].replace("\n", " ")
        print(f"    {preview}{'...' if len(doc) > 300 else ''}")
        print()


if __name__ == "__main__":
    inspect_collection()
