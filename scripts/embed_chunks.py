import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/faiss_indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print("🔍 Loading embedding model: all-MiniLM-L6-v2 ...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Process each chunked JSON
for file in CHUNKS_DIR.glob("*.json"):
    with open(file, "r") as f:
        chunks = json.load(f)

    if not chunks:
        print(f"⚠️ Skipping {file.name}: empty chunk list")
        continue

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Build FAISS index
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    # Save FAISS index
    base = file.stem.replace("_chunks", "")
    index_path = INDEX_DIR / f"{base}.index"
    faiss.write_index(index, str(index_path))
    print(f"✅ Embedded & saved: {index_path.name} ({len(chunks)} chunks)")