import logging
from pathlib import Path
import sys
import os
import json
from typing import List, Dict
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from vectorstore.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_queries() -> List[Dict]:
    """Load test queries with their ground truth section titles."""
    return [
        {
            "query": "How do the models perform in terms of accuracy?",
            "expected_sections": [
                "RQ4: How do models perform with different model sizes?",
                "4 Results",
                "3. Study of systematic uncertainties"
            ]
        },
        {
            "query": "What is the architecture of the system?",
            "expected_sections": [
                "A. Implementation Details",
                "III. METHODOLOGY",
                "2. General truncation for gravity-matter systems"
            ]
        },
        {
            "query": "What are the main limitations of the approach?",
            "expected_sections": [
                "3. Study of systematic uncertainties",
                "IV. THE NECESSITY OF DEGREES OF FREEDOM BEYOND THE STANDARD MODEL IN ASYMPTOTIC SAFETY",
                "2. General truncation for gravity-matter systems"
            ]
        },
        {
            "query": "How does the method compare to previous work?",
            "expected_sections": [
                "III. METHODOLOGY",
                "2. General truncation for gravity-matter systems",
                "3. Study of systematic uncertainties"
            ]
        }
    ]

def evaluate_retrieval(retriever: HybridRetriever, queries: List[Dict], k: int = 5) -> Dict:
    """Evaluate retrieval quality with focus on section matching."""
    metrics = {
        "section_match_rate": 0.0,
        "top_k_accuracy": defaultdict(list),
        "score_distributions": {
            "faiss": [],
            "bm25": [],
            "reranker": [],
            "final": []
        },
        "section_matches": defaultdict(int),
        "total_queries": len(queries)
    }
    
    for query_data in queries:
        query = query_data["query"]
        expected_sections = query_data["expected_sections"]
        
        # Get retrieval results
        results = retriever.retrieve(query, k=k, score_debug=True)
        
        # Track section matches
        section_matches = 0
        for i, result in enumerate(results):
            # Record scores
            metrics["score_distributions"]["faiss"].append(result["faiss_score"])
            metrics["score_distributions"]["bm25"].append(result["bm25_score"])
            if "reranker_score" in result:
                metrics["score_distributions"]["reranker"].append(result["reranker_score"])
            metrics["score_distributions"]["final"].append(result["score"])
            
            # Check section match
            if result["section_title"] in expected_sections:
                section_matches += 1
                metrics["section_matches"][result["section_title"]] += 1
                metrics["top_k_accuracy"][i+1].append(1)
            else:
                metrics["top_k_accuracy"][i+1].append(0)
        
        # Calculate section match rate for this query
        metrics["section_match_rate"] += section_matches / k
    
    # Average out the metrics
    metrics["section_match_rate"] /= len(queries)
    for k in metrics["top_k_accuracy"]:
        metrics["top_k_accuracy"][k] = np.mean(metrics["top_k_accuracy"][k])
    
    # Calculate score statistics
    for score_type in metrics["score_distributions"]:
        scores = metrics["score_distributions"][score_type]
        if scores:
            metrics["score_distributions"][score_type] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
    
    return metrics

def plot_metrics(metrics: Dict, output_dir: Path):
    """Generate plots for the evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot top-k accuracy
    plt.figure(figsize=(10, 6))
    k_values = sorted(metrics["top_k_accuracy"].keys())
    accuracies = [metrics["top_k_accuracy"][k] for k in k_values]
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Top-K Section Match Accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(output_dir / "top_k_accuracy.png")
    plt.close()
    
    # Plot score distributions
    plt.figure(figsize=(12, 6))
    score_types = ["faiss", "bm25", "reranker", "final"]
    means = [metrics["score_distributions"][t]["mean"] for t in score_types]
    stds = [metrics["score_distributions"][t]["std"] for t in score_types]
    
    plt.bar(score_types, means, yerr=stds, capsize=5)
    plt.title("Score Distribution Statistics")
    plt.xlabel("Score Type")
    plt.ylabel("Score Value")
    plt.grid(True)
    plt.savefig(output_dir / "score_distributions.png")
    plt.close()

def main():
    # Initialize retriever
    retriever = HybridRetriever(
        faiss_index_dir="vectorstore/faiss_index",
        chunks_dir="data/final_chunks",
        use_reranker=True,
        section_boost=1.2
    )
    
    # Load test queries
    queries = load_test_queries()
    
    # Run evaluation
    metrics = evaluate_retrieval(retriever, queries)
    
    # Print results
    print("\n📊 Evaluation Results:")
    print(f"Section Match Rate: {metrics['section_match_rate']:.3f}")
    print("\nTop-K Accuracy:")
    for k, acc in sorted(metrics["top_k_accuracy"].items()):
        print(f"  Top-{k}: {acc:.3f}")
    
    print("\nScore Distributions:")
    for score_type, stats in metrics["score_distributions"].items():
        print(f"\n{score_type.upper()} Scores:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Min:  {stats['min']:.3f}")
        print(f"  Max:  {stats['max']:.3f}")
    
    print("\nSection Matches:")
    for section, count in sorted(metrics["section_matches"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {section}: {count}")
    
    # Generate plots
    plot_metrics(metrics, Path("eval/plots"))
    print("\n✨ Plots saved to eval/plots/")

if __name__ == "__main__":
    main() 