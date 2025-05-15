import json
from pathlib import Path
import shutil
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_chunks(mineru_parsed_dir: Path, chunks_output_dir: Path):
    mineru_parsed_dir = Path(mineru_parsed_dir)
    chunks_output_dir = Path(chunks_output_dir)
    chunks_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {mineru_parsed_dir} for magic-pdf outputs...")
    
    paper_dirs = [d for d in mineru_parsed_dir.iterdir() if d.is_dir()] # Top level paper_id dirs
    
    prepared_count = 0
    for paper_dir_level1 in tqdm(paper_dirs, desc="Preparing chunk files"):
        # magic-pdf creates an extra nesting: {paper_id}/{paper_id}/auto/
        paper_id = paper_dir_level1.name
        paper_dir_level2 = paper_dir_level1 / paper_id / "auto"
        
        if not paper_dir_level2.exists():
            logger.warning(f"Could not find 'auto' directory in {paper_dir_level1 / paper_id}")
            continue
            
        source_chunk_file = paper_dir_level2 / f"{paper_id}_content_list.json"
        target_chunk_file = chunks_output_dir / f"{paper_id}_chunks.json"
        
        if not source_chunk_file.exists():
            logger.warning(f"Source chunk file not found: {source_chunk_file}")
            continue
            
        try:
            # We might need to transform the content if the structure is very different.
            # For now, let's assume it has a 'text' field per item or is a list of strings.
            # process_chunks.py expects a list of dicts, each with at least a "text" field.
            # Let's just copy and see if process_chunks.py can handle it.
            shutil.copy(source_chunk_file, target_chunk_file)
            logger.info(f"Copied {source_chunk_file.name} to {target_chunk_file.name}")
            prepared_count += 1
        except Exception as e:
            logger.error(f"Error processing {source_chunk_file}: {e}")
            
    logger.info(f"Successfully prepared {prepared_count} chunk files in {chunks_output_dir}")

if __name__ == "__main__":
    mineru_dir = Path("data/mineru_parsed")
    chunks_dir = Path("data/chunks")
    prepare_chunks(mineru_dir, chunks_dir) 