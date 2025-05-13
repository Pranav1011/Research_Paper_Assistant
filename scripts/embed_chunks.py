import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHUNKS_DIR = Path("data/final_chunks")
INDEX_DIR = Path("vectorstore/faiss_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks(file_path: Path) -> Optional[List[Dict]]:
    """Load and validate chunks from a JSON file."""
    try:
        with open(file_path, "r") as f:
            chunks = json.load(f)
        
        if not chunks:
            logger.warning(f"⚠️ Skipping {file_path.name}: empty chunk list")
            return None
            
        # Validate chunks have text field
        valid_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
        if not valid_chunks:
            logger.warning(f"⚠️ Skipping {file_path.name}: no valid text entries")
            return None
            
        return valid_chunks
    except json.JSONDecodeError:
        logger.error(f"❌ Error: {file_path.name} is not valid JSON")
        return None
    except Exception as e:
        logger.error(f"❌ Error reading {file_path.name}: {str(e)}")
        return None

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create and return a FAISS index for the given embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    return index

def main():
    # Create index directory
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"🔍 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Get all JSON files
    json_files = list(CHUNKS_DIR.glob("*.json"))
    if not json_files:
        logger.error(f"❌ No JSON files found in {CHUNKS_DIR}")
        return
        
    logger.info(f"📚 Found {len(json_files)} JSON files to process")
    
    # Process each file
    for file_path in tqdm(json_files, desc="Processing files"):
        # Load chunks
        chunks = load_chunks(file_path)
        if not chunks:
            continue
            
        try:
            # Extract texts
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"🔄 Generating embeddings for {file_path.name}")
            embeddings = model.encode(texts, show_progress_bar=True)
            
            # Create and save index
            index = create_faiss_index(embeddings)
            index_path = INDEX_DIR / f"{file_path.stem}.index"
            faiss.write_index(index, str(index_path))

            # Save chunk metadata in the same order as embeddings
            meta_path = INDEX_DIR / f"{file_path.stem}_chunks_meta.json"
            with open(meta_path, "w") as meta_f:
                json.dump(chunks, meta_f, indent=2)

            logger.info(f"✅ Saved index: {index_path.name} ({len(chunks)} chunks) and metadata: {meta_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
            continue
    
    logger.info("✨ Index creation complete!")

if __name__ == "__main__":
    main()