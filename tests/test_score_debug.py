import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vectorstore.hybrid_retriever import HybridRetriever

# Configure logging to show all debug information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_score_debug():
    # Initialize retriever
    retriever = HybridRetriever(
        faiss_index_dir="vectorstore/faiss_index",
        chunks_dir="data/final_chunks",
        use_reranker=True,
        faiss_weight=0.7,
        bm25_weight=0.3,
        reranker_weight=0.5
    )
    
    # Test queries with different characteristics
    test_queries = [
        "How do the models perform in terms of accuracy?",  # Technical query
        "What are the main limitations of the approach?",   # Narrative query
        "What is the architecture of the system?"          # Technical query with method terms
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: {query}")
        print(f"{'='*80}")
        
        # Test with score debugging enabled
        results = retriever.retrieve(
            query=query,
            k=3,
            score_debug=True
        )
        
        # Verify that results contain all expected score components
        for i, result in enumerate(results, 1):
            assert 'faiss_score' in result, f"Result {i} missing FAISS score"
            assert 'bm25_score' in result, f"Result {i} missing BM25 score"
            assert 'reranker_score' in result, f"Result {i} missing reranker score"
            assert 'score' in result, f"Result {i} missing final score"
            
        print(f"\nRetrieved {len(results)} results successfully")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    test_score_debug() 