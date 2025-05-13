import json
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import re
from nltk.corpus import stopwords
from nltk import download as nltk_download

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vectorstore.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk_download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def normalize_section_title(title):
    """Normalize section title for comparison."""
    if not title:
        return ""
        
    # Convert to lowercase
    title = title.lower()
    
    # Remove LaTeX commands and special characters
    title = re.sub(r'\\[a-zA-Z]+', '', title)
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # Remove section numbers and common prefixes
    title = re.sub(r'^[ivxIVX]+\.?\s*', '', title)  # Remove roman numerals
    title = re.sub(r'^\d+\.?\s*', '', title)  # Remove numbers
    
    # Remove common prefixes
    prefixes = ['section', 'chapter', 'part', 'appendix']
    for prefix in prefixes:
        title = re.sub(f'^{prefix}\s+', '', title)
    
    # Remove extra whitespace
    title = ' '.join(title.split())
    
    # Remove stopwords
    tokens = [w for w in title.split() if w not in STOPWORDS]
    
    return ' '.join(tokens)

def jaccard_sim(a, b):
    """Calculate Jaccard similarity between two strings."""
    if not a or not b:
        return 0.0
        
    # Normalize both strings
    a = normalize_section_title(a)
    b = normalize_section_title(b)
    
    # Split into words and create sets
    set_a = set(a.split())
    set_b = set(b.split())
    
    if not set_a or not set_b:
        return 0.0
        
    # Calculate Jaccard similarity
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0

class RetrieverEvaluator:
    def __init__(
        self,
        test_queries_path: str = "data/test_queries.json",
        results_dir: str = "evaluation_results"
    ):
        self.test_queries_path = Path(test_queries_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load test queries
        self.test_queries = self._load_test_queries()
        
        # Initialize retriever
        self.retriever = HybridRetriever()
        
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from JSON file."""
        if not self.test_queries_path.exists():
            raise FileNotFoundError(f"Test queries file not found: {self.test_queries_path}")
            
        with open(self.test_queries_path, 'r') as f:
            return json.load(f)
            
    def evaluate_retrieval(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate retrieval quality with various metrics."""
        results = {
            'section_match_rates': [],
            'fuzzy_section_match_rates': [],
            'top_k_accuracy': {k: [] for k in k_values},
            'fuzzy_top_k_accuracy': {k: [] for k in k_values},
            'score_distributions': {
                'faiss': [],
                'bm25': [],
                'combined': []
            },
            'boost_stats': {
                'boosted_count': 0,
                'total_count': 0,
                'boost_factors': []
            }
        }
        
        for query in tqdm(self.test_queries, desc="Evaluating queries"):
            # Get retrieval results
            retrieved = self.retriever.retrieve(
                query['query'],
                k=max(k_values),
                score_debug=True  # Enable score debugging
            )
            
            # Normalize expected and retrieved section titles
            expected_sections = set(normalize_section_title(s) for s in query.get('expected_sections', []))
            retrieved_sections = {normalize_section_title(r['section_title']) for r in retrieved}
            
            # Log section matching for debugging
            logger.info(f"\nQuery: {query['query']}")
            logger.info(f"Expected sections: {expected_sections}")
            logger.info(f"Retrieved sections: {retrieved_sections}")
            
            # Exact match rate
            section_match_rate = len(expected_sections & retrieved_sections) / len(expected_sections) if expected_sections else 0
            results['section_match_rates'].append(section_match_rate)
            
            # Fuzzy match rate (Jaccard > 0.5)
            fuzzy_matches = 0
            for es in expected_sections:
                for rs in retrieved_sections:
                    sim = jaccard_sim(es, rs)
                    if sim > 0.5:
                        fuzzy_matches += 1
                        logger.info(f"Fuzzy match: {es} ~ {rs} (sim: {sim:.2f})")
                        break
            fuzzy_section_match_rate = fuzzy_matches / len(expected_sections) if expected_sections else 0
            results['fuzzy_section_match_rates'].append(fuzzy_section_match_rate)
            
            # Top-k accuracy (exact and fuzzy)
            for k in k_values:
                top_k_sections = {normalize_section_title(r['section_title']) for r in retrieved[:k]}
                accuracy = len(expected_sections & top_k_sections) / len(expected_sections) if expected_sections else 0
                results['top_k_accuracy'][k].append(accuracy)
                
                # Fuzzy
                fuzzy_matches_k = 0
                for es in expected_sections:
                    for rs in top_k_sections:
                        if jaccard_sim(es, rs) > 0.5:
                            fuzzy_matches_k += 1
                            break
                fuzzy_accuracy = fuzzy_matches_k / len(expected_sections) if expected_sections else 0
                results['fuzzy_top_k_accuracy'][k].append(fuzzy_accuracy)
            
            # Collect score distributions
            for result in retrieved:
                results['score_distributions']['faiss'].append(result['faiss_score'])
                results['score_distributions']['bm25'].append(result['bm25_score'])
                results['score_distributions']['combined'].append(result['score'])
                
                # Collect boost statistics
                if result.get('boosted', False):
                    results['boost_stats']['boosted_count'] += 1
                    results['boost_stats']['boost_factors'].append(result.get('boost_factor', 1.0))
                results['boost_stats']['total_count'] += 1
        
        return results
    
    def plot_results(self, results: Dict[str, Any]):
        """Generate plots for evaluation results."""
        # Plot section match rate distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['section_match_rates'], bins=20, alpha=0.5, label='Exact')
        plt.hist(results['fuzzy_section_match_rates'], bins=20, alpha=0.5, label='Fuzzy (Jaccard > 0.5)')
        plt.title('Section Match Rate Distribution')
        plt.xlabel('Match Rate')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(self.results_dir / 'section_match_rates.png')
        plt.close()
        
        # Plot top-k accuracy
        plt.figure(figsize=(10, 6))
        k_values = sorted(results['top_k_accuracy'].keys())
        accuracies = [np.mean(results['top_k_accuracy'][k]) for k in k_values]
        fuzzy_accuracies = [np.mean(results['fuzzy_top_k_accuracy'][k]) for k in k_values]
        plt.plot(k_values, accuracies, marker='o', label='Exact')
        plt.plot(k_values, fuzzy_accuracies, marker='x', label='Fuzzy (Jaccard > 0.5)')
        plt.title('Top-K Accuracy')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.results_dir / 'top_k_accuracy.png')
        plt.close()
        
        # Plot score distributions
        plt.figure(figsize=(12, 6))
        for score_type in ['faiss', 'bm25', 'combined']:
            plt.hist(results['score_distributions'][score_type], 
                    bins=20, alpha=0.5, label=score_type)
        plt.title('Score Distributions')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(self.results_dir / 'score_distributions.png')
        plt.close()
        
        # Plot boost factor distribution
        if results['boost_stats']['boost_factors']:
            plt.figure(figsize=(10, 6))
            plt.hist(results['boost_stats']['boost_factors'], bins=20)
            plt.title('Boost Factor Distribution')
            plt.xlabel('Boost Factor')
            plt.ylabel('Frequency')
            plt.savefig(self.results_dir / 'boost_factors.png')
            plt.close()
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file."""
        # Calculate summary statistics
        summary = {
            'mean_section_match_rate': float(np.mean(results['section_match_rates'])),
            'mean_fuzzy_section_match_rate': float(np.mean(results['fuzzy_section_match_rates'])),
            'top_k_accuracy': {
                int(k): float(np.mean(accuracies)) 
                for k, accuracies in results['top_k_accuracy'].items()
            },
            'fuzzy_top_k_accuracy': {
                int(k): float(np.mean(accuracies)) 
                for k, accuracies in results['fuzzy_top_k_accuracy'].items()
            },
            'score_stats': {
                score_type: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
                for score_type, scores in results['score_distributions'].items()
            },
            'boost_stats': {
                'boost_rate': float(results['boost_stats']['boosted_count']) / float(results['boost_stats']['total_count']),
                'mean_boost_factor': float(np.mean(results['boost_stats']['boost_factors'])) if results['boost_stats']['boost_factors'] else 0.0
            }
        }
        
        # Convert all numpy types in detailed_results to native Python types
        def convert(obj):
            if isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        detailed_results = convert(results)
        
        # Save detailed results
        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': detailed_results
            }, f, indent=2)
        
        # Print summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"Mean Section Match Rate: {summary['mean_section_match_rate']:.3f}")
        logger.info(f"Mean Fuzzy Section Match Rate: {summary['mean_fuzzy_section_match_rate']:.3f}")
        logger.info("\nTop-K Accuracy:")
        for k, acc in summary['top_k_accuracy'].items():
            logger.info(f"  k={k}: {acc:.3f}")
        logger.info("\nFuzzy Top-K Accuracy:")
        for k, acc in summary['fuzzy_top_k_accuracy'].items():
            logger.info(f"  k={k}: {acc:.3f}")
        logger.info("\nScore Statistics:")
        for score_type, stats in summary['score_stats'].items():
            logger.info(f"\n{score_type.upper()} Scores:")
            logger.info(f"  Mean: {stats['mean']:.3f}")
            logger.info(f"  Std:  {stats['std']:.3f}")
            logger.info(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        logger.info("\nBoost Statistics:")
        logger.info(f"  Boost Rate: {summary['boost_stats']['boost_rate']:.3f}")
        logger.info(f"  Mean Boost Factor: {summary['boost_stats']['mean_boost_factor']:.3f}")

def main():
    evaluator = RetrieverEvaluator()
    results = evaluator.evaluate_retrieval()
    evaluator.plot_results(results)
    evaluator.save_results(results)

if __name__ == "__main__":
    main() 