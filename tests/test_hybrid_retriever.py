import logging
from pathlib import Path
import sys
import os

# Add the parent directory to Python path to import from vectorstore
sys.path.append(str(Path(__file__).parent.parent))

from vectorstore.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length and add ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def print_results(results):
    """Print search results in a readable format."""
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"Result {i} (Score: {result['score']:.3f})")
        print(f"Section: {result['section_title']}")
        print(f"Source: {result['source_file']}")
        print(f"FAISS Score: {result['faiss_score']:.3f}")
        print(f"BM25 Score: {result['bm25_score']:.3f}")
        if 'reranker_score' in result:
            print(f"Reranker Score: {result['reranker_score']:.3f}")
        if result.get('boosted', False):
            print("✨ Section Boost Applied")
        print("\nText:")
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
        print(f"{'='*80}\n")

def main():
    # Initialize retriever
    retriever = HybridRetriever(
        faiss_index_dir="vectorstore/faiss_index",
        chunks_dir="data/final_chunks",
        use_reranker=True,
        section_boost=1.2
    )
    
    # Test queries
    test_queries = [
        "How do the models perform in terms of accuracy?",
        "What is the architecture of the system?",
        "What are the main limitations of the approach?",
        "How does the method compare to previous work?"
    ]
    
    # Test with different configurations
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        
        # Test 1: Basic retrieval
        print("\n1️⃣ Basic retrieval (top 3 results):")
        results = retriever.retrieve(query, k=3)
        print_results(results)
        
        # Test 2: With section filter
        print("\n2️⃣ Filtered by 'Results' section:")
        results = retriever.retrieve(
            query, 
            k=3,
            metadata_filters={'section_title': 'Results'}
        )
        print_results(results)
        
        # Test 3: With method/architecture filter
        print("\n3️⃣ Filtered by 'Method' or 'Architecture' section:")
        results = retriever.retrieve(
            query,
            k=3,
            metadata_filters={'section_title': ['Method', 'Architecture']}
        )
        print_results(results)

if __name__ == "__main__":
    main() 