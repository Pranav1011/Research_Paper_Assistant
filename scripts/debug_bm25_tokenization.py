import sys
import os
from pathlib import Path
import nltk

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vectorstore.hybrid_retriever import HybridRetriever

nltk.download('punkt')

retriever = HybridRetriever(use_reranker=False, use_query_expansion=False, faiss_weight=0.0, bm25_weight=1.0)

# Example queries
queries = [
    "What is the architecture of the system?",
    "How does the retriever handle query expansion?",
    "Explain the tokenization process used in BM25 scoring"
]

print("\n--- Tokenized Queries ---")
for q in queries:
    print(f"Query: {q}")
    print(f"Tokens: {retriever._tokenize_text(q)}\n")

print("\n--- Tokenized Corpus Chunks (first 2) ---")
for i, chunk in enumerate(retriever.chunks[:2]):
    text = chunk.get('text', '')
    section = chunk.get('section_title', '')
    combined = f"{section} {text}"
    print(f"Chunk {i+1} Section: {section}")
    print(f"Chunk {i+1} Text: {text[:100]}...")
    print(f"Tokens: {retriever._tokenize_text(combined)}\n")
    if i == 1:
        break 