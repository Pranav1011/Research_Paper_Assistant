from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import os
import re
from pathlib import Path
from nltk.corpus import wordnet
import nltk
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRetriever:
    # Keywords that indicate important sections
    IMPORTANT_SECTION_KEYWORDS = {
        'model', 'architecture', 'framework', 'method', 'approach',
        'implementation', 'system', 'algorithm', 'technique', 'design'
    }
    
    def __init__(
        self,
        faiss_index_dir: str = "vectorstore/faiss_index",
        chunks_dir: str = "data/final_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_name: str = "BAAI/bge-reranker-base",
        faiss_weight: float = 0.7,
        bm25_weight: float = 0.3,
        section_boost: float = 1.2,
        use_reranker: bool = True,
        use_query_expansion: bool = True,
        reranker_weight: float = 0.5
    ):
        """
        Initialize the hybrid retriever combining FAISS and BM25.
        
        Args:
            faiss_index_dir: Directory containing FAISS index files
            chunks_dir: Directory containing chunk files
            model_name: Name of the sentence transformer model
            reranker_name: Name of the reranker model
            faiss_weight: Weight for FAISS scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            section_boost: Score multiplier for important sections
            use_reranker: Whether to use the reranker for final results
            use_query_expansion: Whether to use query expansion with synonyms
            reranker_weight: Weight for reranker score in final score (0-1, only used if reranker is enabled)
        """
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.section_boost = section_boost
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        self.reranker_weight = reranker_weight
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        # Initialize paths
        self.faiss_index_dir = Path(faiss_index_dir)
        self.chunks_dir = Path(chunks_dir)
        
        # Load models
        logger.info(f"🔍 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        if self.use_reranker:
            logger.info(f"🔍 Loading reranker model: {reranker_name}")
            self.reranker = CrossEncoder(reranker_name)
        
        # Load chunks and prepare indices
        self.chunks = self._load_chunks()
        self.faiss_indices = self._load_faiss_indices()
        self.bm25 = self._prepare_bm25(self.chunks)
        
        logger.info(f"✅ Loaded {len(self.chunks)} chunks and {len(self.faiss_indices)} FAISS indices")
        
        # Initialize cache
        self._cache = {}
        self._cache_size = 1000  # Maximum number of cached results
        
        # Initialize section title embeddings
        self.section_title_embeddings = {}
        self._prepare_section_title_embeddings()
    
    def save_index(self):
        """Save the FAISS index to disk."""
        for paper_id, index in self.faiss_indices.items():
            faiss.write_index(index, str(self.faiss_index_dir / f"{paper_id}.index"))
    
    def add_to_index(self, texts: List[str]):
        """
        Add new texts to the FAISS index.
        
        Args:
            texts: List of text chunks to add to the index
        """
        if not texts:
            print("⚠️ Warning: No texts provided to add to index")
            return
            
        # Encode texts
        embeddings = self.model.encode(texts)
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        for paper_id, index in self.faiss_indices.items():
            index.add(embeddings)
        
        # Save updated indices
        self.save_index()
        
        # Update BM25 index
        self.chunks.extend([{'text': text} for text in texts])
        self.bm25 = self._prepare_bm25(self.chunks)
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunk files from the chunks directory."""
        chunks = []
        
        if not self.chunks_dir.exists():
            logger.error(f"❌ Chunks directory {self.chunks_dir} does not exist")
            return chunks
            
        for chunk_file in tqdm(list(self.chunks_dir.glob("*.json")), desc="Loading chunks"):
            try:
                with open(chunk_file, 'r') as f:
                    file_chunks = json.load(f)
                    if isinstance(file_chunks, list):
                        chunks.extend(file_chunks)
                    elif isinstance(file_chunks, dict):
                        chunks.append(file_chunks)
            except Exception as e:
                logger.error(f"❌ Error reading {chunk_file}: {str(e)}")
                
        return chunks
    
    def _load_faiss_indices(self) -> Dict[str, faiss.Index]:
        """Load all FAISS indices from the index directory."""
        indices = {}
        self.chunk_metadata = {}
        
        if not self.faiss_index_dir.exists():
            logger.error(f"❌ FAISS index directory {self.faiss_index_dir} does not exist")
            return indices
            
        for index_file in self.faiss_index_dir.glob("*.index"):
            try:
                indices[index_file.stem] = faiss.read_index(str(index_file))
                # Load corresponding chunk metadata
                meta_path = self.faiss_index_dir / f"{index_file.stem}_chunks_meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as meta_f:
                        self.chunk_metadata[index_file.stem] = json.load(meta_f)
            except Exception as e:
                logger.error(f"❌ Error loading index {index_file}: {str(e)}")
                
        return indices
    
    def _prepare_bm25(self, chunks: List[Dict[str, Any]]) -> BM25Okapi:
        """Prepare BM25 index from chunks."""
        # Combine section titles and text with higher weight for titles
        documents = []
        for chunk in chunks:
            # Process section title
            title = chunk.get('section_title', '')
            title_tokens = self._tokenize_text(title)
            
            # Process text
            text = chunk.get('text', '')
            text_tokens = self._tokenize_text(text)
            
            # Combine with higher weight for title (repeat title tokens)
            combined_tokens = title_tokens * 3 + text_tokens
            
            documents.append(combined_tokens)  # <-- pass token list, not joined string
        
        # Create BM25 index
        self.bm25 = BM25Okapi(documents)
        return self.bm25

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 scoring."""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove LaTeX commands and special characters
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove section numbers and common prefixes
        text = re.sub(r'^[ivxIVX]+\.?\s*', '', text)  # Remove roman numerals
        text = re.sub(r'^\d+\.?\s*', '', text)  # Remove numbers
        
        # Remove common prefixes
        prefixes = ['section', 'chapter', 'part', 'appendix']
        for prefix in prefixes:
            text = re.sub(f'^{prefix}\s+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize using NLTK
        tokens = nltk.word_tokenize(text)
        
        # Filter tokens
        tokens = [
            token for token in tokens
            if len(token) > 1  # Remove single characters
            and token not in {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
            }  # Remove stopwords
            and not token.isdigit()  # Remove pure numbers
            and not re.match(r'^[ivxIVX]+$', token)  # Remove roman numerals
        ]
        
        return tokens

    def _get_faiss_scores(self, query_emb, k=20):
        """Retrieve top k results from all FAISS indices and return (chunk_id, score) pairs."""
        all_scores = []
        # Reshape query_emb to a 2D array (1, dim)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        for paper_id, index in self.faiss_indices.items():
            if paper_id not in self.chunk_metadata:
                continue
            D, I = index.search(query_emb, k)
            # D: distances (lower is better), I: indices
            for idx, distances in enumerate(D):
                similarities = 1 / (1 + distances)  # Convert L2 distance to similarity
                # Min-max normalize similarities
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)
                if max_sim > min_sim:
                    norm_similarities = (similarities - min_sim) / (max_sim - min_sim)
                else:
                    norm_similarities = np.ones_like(similarities) * 0.5
                for j, chunk_idx in enumerate(I[idx]):
                    if 0 <= chunk_idx < len(self.chunk_metadata[paper_id]):
                        chunk_id = f"{paper_id}_{chunk_idx}"
                        score = float(norm_similarities[j])
                        all_scores.append((chunk_id, score))
        return all_scores
    
    def _get_bm25_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query."""
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        # Get raw scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores to [0, 1] range
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _prepare_section_title_embeddings(self):
        """Prepare embeddings for section titles to enable semantic matching."""
        unique_sections = set()
        for chunk in self.chunks:
            if isinstance(chunk, dict):
                section_title = chunk.get('section_title', '')
                if section_title:
                    unique_sections.add(section_title.lower())
        
        if unique_sections:
            section_titles = list(unique_sections)
            embeddings = self.model.encode(section_titles)
            self.section_title_embeddings = {
                title: emb for title, emb in zip(section_titles, embeddings)
            }

    def _calculate_section_similarity(self, query: str, section_title: str) -> float:
        """Calculate semantic similarity between query and section title."""
        if not section_title or not query:
            return 0.0
            
        # Get section title embedding
        section_title_lower = section_title.lower()
        if section_title_lower not in self.section_title_embeddings:
            return 0.0
            
        # Get query embedding
        query_emb = self.model.encode([query])[0]
        section_emb = self.section_title_embeddings[section_title_lower]
        
        # Calculate cosine similarity
        similarity = np.dot(query_emb, section_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(section_emb)
        )
        
        return float(similarity)

    def _apply_section_boost(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply score boost to chunks from important sections using semantic matching."""
        for result in results:
            section_title = result.get('section_title', '')
            if not section_title:
                result['boosted'] = False
                continue
                
            # Calculate semantic similarity
            semantic_similarity = self._calculate_section_similarity(query, section_title)
            
            # Check for keyword matches
            keyword_match = any(
                keyword in section_title.lower() 
                for keyword in self.IMPORTANT_SECTION_KEYWORDS
            )
            
            # Calculate boost factor based on both semantic similarity and keywords
            boost_factor = 1.0
            if semantic_similarity > 0.7:  # High semantic similarity
                boost_factor = self.section_boost * 1.2
            elif semantic_similarity > 0.5:  # Medium semantic similarity
                boost_factor = self.section_boost * 1.1
            elif keyword_match:  # Keyword match only
                boost_factor = self.section_boost
            
            # Apply boost
            if boost_factor > 1.0:
                result['score'] *= boost_factor
                result['boosted'] = True
                result['boost_factor'] = boost_factor
                result['semantic_similarity'] = semantic_similarity
            else:
                result['boosted'] = False
                
        return results
    
    def _apply_metadata_filters(
        self, 
        results: List[Dict[str, Any]], 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter results based on metadata criteria."""
        if not filters:
            return results
            
        filtered_results = []
        for result in results:
            matches = True
            for key, value in filters.items():
                if key not in result:
                    matches = False
                    break
                # Case-insensitive comparison for section titles
                if key == 'section_title':
                    if isinstance(value, list):
                        # Check if any of the values in the list match
                        if not any(v.lower() in result[key].lower() for v in value):
                            matches = False
                            break
                    else:
                        # Single value comparison
                        if value.lower() not in result[key].lower():
                            matches = False
                            break
                # Exact match for other fields
                elif result[key] != value:
                    matches = False
                    break
            if matches:
                filtered_results.append(result)
                
        return filtered_results
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query by normalizing text and removing special characters."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        query = re.sub(r'[^a-z0-9\s.,?!]', '', query)
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        return query

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms using WordNet."""
        if not self.use_query_expansion:
            return query
            
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            # Get synonyms from WordNet
            synsets = wordnet.synsets(word)
            if synsets:
                # Add up to 2 synonyms per word
                synonyms = [lemma.name() for synset in synsets[:2] 
                          for lemma in synset.lemmas() 
                          if lemma.name() != word]
                expanded_words.extend(synonyms[:2])
        
        return ' '.join(expanded_words)

    def _calculate_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """Calculate dynamic weights based on query characteristics and content analysis."""
        # Default weights
        faiss_w = self.faiss_weight
        bm25_w = self.bm25_weight
        
        # Technical terms that might benefit from BM25
        technical_terms = {
            'api', 'function', 'method', 'class', 'interface', 
            'protocol', 'algorithm', 'implementation', 'code',
            'variable', 'parameter', 'return', 'type', 'struct',
            'enum', 'constant', 'macro', 'define', 'include'
        }
        
        # Semantic terms that might benefit from FAISS
        semantic_terms = {
            'how', 'what', 'why', 'when', 'where', 'explain',
            'describe', 'compare', 'difference', 'similar',
            'example', 'use case', 'scenario', 'approach',
            'solution', 'problem', 'issue', 'challenge'
        }
        
        # Count term occurrences
        query_lower = query.lower()
        tech_count = sum(1 for term in technical_terms if term in query_lower)
        semantic_count = sum(1 for term in semantic_terms if term in query_lower)
        
        # Calculate query length factor
        query_length = len(query.split())
        length_factor = min(1.0, query_length / 10)  # Normalize by typical query length
        
        # Calculate adjustments based on term counts and query length
        if tech_count > 0 or semantic_count > 0:
            total_terms = tech_count + semantic_count
            tech_ratio = tech_count / total_terms
            semantic_ratio = semantic_count / total_terms
            
            # Adjust weights based on term ratios and query length
            adjustment = min(0.3, (tech_ratio - semantic_ratio) * length_factor)
            faiss_w -= adjustment
            bm25_w += adjustment
            
            # Ensure weights stay within reasonable bounds
            faiss_w = max(0.3, min(0.8, faiss_w))
            bm25_w = max(0.2, min(0.7, bm25_w))
            
            # Normalize weights to sum to 1
            total = faiss_w + bm25_w
            faiss_w /= total
            bm25_w /= total
        
        return faiss_w, bm25_w

    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
        score_debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k results using hybrid retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            metadata_filters: Optional filters for metadata fields
            score_debug: Whether to print detailed score information for each result
            
        Returns:
            List of dictionaries containing retrieved chunks with scores
        """
        logger.info(f"🔍 Processing query: {query}")
        
        # Check cache first
        cache_key = f"{query}_{k}_{str(metadata_filters)}_{score_debug}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Preprocess and expand query
        processed_query = self._preprocess_query(query)
        expanded_query = self._expand_query(processed_query)
        
        # Calculate dynamic weights
        faiss_w, bm25_w = self._calculate_dynamic_weights(processed_query)
        
        if score_debug:
            logger.info(f"📊 Using weights - FAISS: {faiss_w:.2f}, BM25: {bm25_w:.2f}")
            logger.info(f"BM25 tokenized query: {self._tokenize_text(processed_query)}")
            logger.info(f"BM25 tokenized chunk example: {self._tokenize_text((self.chunks[0].get('section_title','') + ' ' + self.chunks[0].get('text','')).lower())}")
            
        # Get scores from both retrievers
        query_emb = self.model.encode([expanded_query])[0]
        faiss_results = self._get_faiss_scores(query_emb, k=20)
        bm25_results = self._get_bm25_scores(processed_query)
        
        # Normalize scores
        faiss_scores = self._normalize_scores([score for _, score in faiss_results])
        bm25_scores = self._normalize_scores(bm25_results)
        
        # Combine scores with dynamic weights
        weights = self._calculate_dynamic_weights(query)
        faiss_w, bm25_w = weights
        combined_scores = [
            faiss_w * faiss_scores[i] + bm25_w * bm25_scores[i]
            for i in range(len(faiss_scores))
        ]
        
        # Combine scores
        combined_results = []
        seen_indices = set()
        
        for idx in range(len(faiss_scores)):
            if idx in seen_indices:
                continue
                
            # Get scores from both retrievers
            faiss_score = faiss_scores[idx]
            bm25_score = bm25_scores[idx]
            
            # Calculate combined score
            combined_score = combined_scores[idx]
            
            # Get chunk data
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                result = {
                    'text': chunk.get('text', ''),
                    'section_title': chunk.get('section_title', 'Unknown'),
                    'source_file': chunk.get('source_file', ''),
                    'score': combined_score,
                    'faiss_score': faiss_score,
                    'bm25_score': bm25_score
                }
                combined_results.append(result)
                seen_indices.add(idx)
            else:
                paper_id, chunk_idx = idx.rsplit('_', 1)
                chunk_idx = int(chunk_idx)
                if paper_id in self.chunk_metadata and chunk_idx < len(self.chunk_metadata[paper_id]):
                    chunk = self.chunk_metadata[paper_id][chunk_idx]
                    result = {
                        'text': chunk.get('text', ''),
                        'section_title': chunk.get('section_title', 'Unknown'),
                        'source_file': chunk.get('source_file', ''),
                        'score': combined_score,
                        'faiss_score': faiss_score,
                        'bm25_score': bm25_score
                    }
                    combined_results.append(result)
                    seen_indices.add(idx)
        
        # Apply section boost
        combined_results = self._apply_section_boost(combined_results, query)
        
        # Apply metadata filters if provided
        if metadata_filters:
            combined_results = self._apply_metadata_filters(combined_results, metadata_filters)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply reranker if enabled
        if self.use_reranker and combined_results:
            logger.info("🔄 Applying reranker to top results")
            reranker_input = [(query, result['text']) for result in combined_results[:20]]
            reranker_scores = self.reranker.predict(reranker_input)
            
            # Update scores with reranker scores
            for result, score in zip(combined_results[:20], reranker_scores):
                result['reranker_score'] = float(score)
                # Weighted combination of hybrid and reranker scores
                result['score'] = (1 - self.reranker_weight) * result['score'] + self.reranker_weight * float(score)
        
        # Print debug information if enabled
        if score_debug:
            logger.info("\n📊 Score Debug Information:")
            for i, result in enumerate(combined_results[:k], 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Section: {result['section_title']}")
                logger.info(f"Source: {result['source_file']}")
                logger.info(f"FAISS Score: {result['faiss_score']:.3f}")
                logger.info(f"BM25 Score: {result['bm25_score']:.3f}")
                if 'reranker_score' in result:
                    logger.info(f"Reranker Score: {result['reranker_score']:.3f}")
                logger.info(f"Final Score: {result['score']:.3f}")
                if result.get('boosted', False):
                    logger.info("✨ Section Boost Applied")
                logger.info(f"Text Preview: {result['text'][:200]}...")
        
        # Cache results
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry if cache is full
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = combined_results[:k]
        
        # Log retrieval statistics
        logger.info(f"✅ Retrieved {len(combined_results)} results")
        if combined_results:
            logger.info(f"📊 Score ranges: FAISS [{min(r['faiss_score'] for r in combined_results):.2f}, "
                       f"{max(r['faiss_score'] for r in combined_results):.2f}], "
                       f"BM25 [{min(r['bm25_score'] for r in combined_results):.2f}, "
                       f"{max(r['bm25_score'] for r in combined_results):.2f}]")
        
        return combined_results[:k] 