import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Config ---
INDEX_DIR = Path("data/faiss_indexes")
CHUNKS_DIR = Path("data/chunks")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

def load_chunks(file_id):
    path = CHUNKS_DIR / f"{file_id}_chunks.json"
    with open(path, "r") as f:
        return json.load(f)

def search_index(file_id, query, model):
    # Load FAISS index
    index_path = INDEX_DIR / f"{file_id}.index"
    if not index_path.exists():
        print(f"❌ No index found for {file_id}")
        return []

    index = faiss.read_index(str(index_path))

    # Load corresponding chunks
    chunks = load_chunks(file_id)

    # Encode query
    q_vec = model.encode([query])
    q_vec = np.array(q_vec).astype("float32")

    # Search top-k
    distances, indices = index.search(q_vec, TOP_K)
    return [chunks[i] for i in indices[0]]

if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)

    # List available index files
    index_files = [f for f in INDEX_DIR.glob("*.index")]
    print(f"🔎 Found {len(index_files)} index files")

    for index_file in index_files:
        file_id = index_file.stem
        print(f"\n📘 Querying: {file_id}")
        query = input("🔍 Enter your test query: ")
        results = search_index(file_id, query, model)
        for i, r in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(r["text"])