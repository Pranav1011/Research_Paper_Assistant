import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vectorstore.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(use_reranker=False, use_query_expansion=False, faiss_weight=0.0, bm25_weight=1.0)

queries = [
    "Explain the tokenization process used in BM25 scoring",  # technical
    "What is the purpose of section boosting in the retriever?"  # narrative
]

for q in queries:
    print(f"\n=== BM25-Only Retrieval for Query: '{q}' ===")
    bm25_results = retriever._get_bm25_scores(q, k=5)
    for idx, score in bm25_results:
        if idx.startswith('bm25_'):
            chunk_idx = int(idx.split('_')[1])
            chunk = retriever.chunks[chunk_idx]
            section = chunk.get('section_title', '')
            text = chunk.get('text', '')
            print(f"Section: {section}")
            print(f"Score: {score}")
            print(f"Text: {text[:120]}...\n") 