import json
import re
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def is_symbol_heavy(text: str, threshold: float = 0.6) -> bool:
    """Check if text contains too many symbols."""
    if not text.strip():
        return True
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return (symbols / max(len(text), 1)) > threshold

def get_section_embeddings(md_content: str) -> List[Tuple[str, np.ndarray]]:
    """
    Extract all section titles from markdown content and compute their embeddings.
    
    Args:
        md_content: Full content of the markdown file
        
    Returns:
        List of (section_title, embedding) tuples
    """
    # Split markdown content into lines
    lines = md_content.split('\n')
    
    # Find all headers and their positions
    section_titles = []
    for line in lines:
        if re.match(r'^#{1,3}\s+', line):
            # Extract the header text
            title = re.sub(r'^#{1,3}\s+', '', line).strip()
            section_titles.append(title)
    
    if not section_titles:
        return []
    
    # Compute embeddings for all section titles
    embeddings = model.encode(section_titles)
    
    return list(zip(section_titles, embeddings))

def find_section_title_semantic(chunk_text: str, section_embeddings: List[Tuple[str, np.ndarray]], 
                              similarity_threshold: float = 0.3) -> str:
    """
    Find the most semantically relevant section title for a chunk using embeddings.
    
    Args:
        chunk_text: Text content of the chunk
        section_embeddings: List of (section_title, embedding) tuples
        similarity_threshold: Minimum cosine similarity to consider a match
        
    Returns:
        Section title if found, "Unknown" otherwise
    """
    if not section_embeddings:
        return "Unknown"
    
    # Compute embedding for the chunk
    chunk_embedding = model.encode(chunk_text)
    
    # Calculate cosine similarities with all section titles
    similarities = []
    for title, embedding in section_embeddings:
        similarity = np.dot(chunk_embedding, embedding) / (
            np.linalg.norm(chunk_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((title, similarity))
    
    # Sort similarities for debugging
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Find the title with highest similarity
    best_title, best_similarity = similarities[0]
    
    # Debug logging
    print(f"\n🔍 Section Title Matching:")
    print(f"Top 3 matches:")
    for title, sim in similarities[:3]:
        print(f"  - {title}: {sim:.3f}")
    print(f"Threshold: {similarity_threshold}")
    print(f"Best match: {best_title} ({best_similarity:.3f})")
    
    # Return the best matching title if similarity is above threshold
    if best_similarity >= similarity_threshold:
        return best_title
    
    print("❌ No match above threshold")
    return "Unknown"

def clean_chunk(chunk: Dict) -> Optional[Dict]:
    """
    Clean a single chunk by removing empty, short, or symbol-heavy content.
    
    Args:
        chunk: Dictionary containing chunk data
        
    Returns:
        Cleaned chunk if valid, None otherwise
    """
    text = chunk.get("text", "").strip()
    
    if not text:
        return None
    if len(text.split()) < 5:
        return None
    if is_symbol_heavy(text):
        return None
        
    return chunk

def process_chunks(chunks_dir: Path, mineru_dir: Path, output_dir: Path) -> None:
    """
    Process all chunk files, clean them, and enrich them with section titles.
    
    Args:
        chunks_dir: Directory containing chunk files
        mineru_dir: Directory containing mineru parsed files
        output_dir: Directory to save processed chunks
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug: Print directory contents
    print(f"\n📁 Input chunks directory: {chunks_dir}")
    print("Found files:")
    for f in chunks_dir.glob("*_chunks.json"):
        print(f"  - {f.name}")
    
    # Process each chunk file
    for chunk_file in chunks_dir.glob("*_chunks.json"):
        print(f"\n{'='*80}")
        print(f"Processing {chunk_file.name}...")
        print(f"{'='*80}")
        
        # Get corresponding markdown file
        paper_id = chunk_file.stem.replace("_chunks", "")
        md_file = mineru_dir / paper_id / "auto" / f"{paper_id}.md"
        
        if not md_file.exists():
            print(f"⚠️ Warning: Markdown file not found: {md_file}")
            continue
        
        # Read markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Pre-compute section embeddings for this document
        section_embeddings = get_section_embeddings(md_content)
        print(f"\n📚 Found {len(section_embeddings)} section titles:")
        for title, _ in section_embeddings:
            print(f"  - {title}")
        
        # Read and process chunks
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        processed_chunks = []
        seen_texts = set()
        
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                print(f"⚠️ Warning: Invalid chunk format in {chunk_file.name}")
                continue
            
            # Clean the chunk
            cleaned_chunk = clean_chunk(chunk)
            if not cleaned_chunk:
                continue
            
            # Get chunk text
            text = cleaned_chunk.get('text', '').strip()
            
            # Skip duplicates
            if text in seen_texts:
                continue
            seen_texts.add(text)
            
            print(f"\n📄 Processing chunk {i+1}/{len(chunks)}")
            print(f"Text preview: {text[:100]}...")
            
            # Find section title using semantic similarity
            section_title = find_section_title_semantic(text, section_embeddings)
            
            # Create processed chunk while preserving all original fields
            processed_chunk = {
                'chunk_id': i,  # Keep original chunk_id format
                'text': text,
                'title': cleaned_chunk.get('title', ''),  # Preserve the original title
                'section_title': section_title,
                'source_file': f"{paper_id}.md",  # Match original format
                'language': cleaned_chunk.get('language', 'en'),  # Preserve language
                'is_english': cleaned_chunk.get('is_english', True),  # Preserve is_english
                'length': len(text)  # Update length based on text
            }
            
            # Preserve any other fields from the original chunk
            for key, value in cleaned_chunk.items():
                if key not in processed_chunk:
                    processed_chunk[key] = value
            
            processed_chunks.append(processed_chunk)
        
        # Save processed chunks
        output_file = output_dir / f"{paper_id}_final_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Processed and saved: {output_file.name} ({len(processed_chunks)} chunks)")
        print(f"{'='*80}\n")

def main():
    # Define paths
    base_dir = Path("data")
    chunks_dir = base_dir / "chunks"
    mineru_dir = base_dir / "mineru_parsed"
    output_dir = base_dir / "final_chunks"
    
    # Process chunks
    process_chunks(chunks_dir, mineru_dir, output_dir)
    print("\n✨ Chunk processing complete!")

if __name__ == "__main__":
    main() 