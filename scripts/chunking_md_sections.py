import os
import json
from pathlib import Path

CHUNK_DIR = Path("data/chunks")
INPUT_GLOB = Path("data/mineru_parsed")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

def split_md_by_sections(md_text):
    """Splits markdown text into sections based on headings."""
    chunks = []
    current_chunk = {
        "title": None,
        "section_title": None,
        "content": []
    }
    current_section = None

    for line in md_text.split("\n"):
        if line.startswith("#"):
            # Save current chunk if it has content
            if current_chunk["content"]:
                chunks.append(current_chunk)
                current_chunk = {
                    "title": None,
                    "section_title": current_section,
                    "content": []
                }
            
            # Update section title for any header level (#, ##, ###)
            current_section = line.strip("#").strip()
            current_chunk["section_title"] = current_section
            current_chunk["title"] = line.strip()
        else:
            current_chunk["content"].append(line)

    if current_chunk["content"]:
        chunks.append(current_chunk)

    # Final cleanup
    cleaned_chunks = []
    for chunk in chunks:
        text = "\n".join(chunk["content"]).strip()
        if text:
            cleaned_chunks.append({
                "title": chunk["title"],
                "section_title": chunk["section_title"],
                "text": text,
                "language": "en",  # Default to English
                "is_english": True,  # Default to English
                "length": len(text)
            })
    return cleaned_chunks

def process_all_markdown_files():
    md_files = list(INPUT_GLOB.glob("*/auto/*.md"))
    print(f"📄 Found {len(md_files)} markdown files")

    for md_file in md_files:
        try:
            with open(md_file, "r") as f:
                md_text = f.read()

            chunks = split_md_by_sections(md_text)
            output_file = CHUNK_DIR / f"{md_file.stem}_chunks.json"

            with open(output_file, "w") as f:
                json.dump(chunks, f, indent=2)

            print(f"✅ Chunked and saved: {output_file.name} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"❌ Failed on {md_file.name}: {e}")

if __name__ == "__main__":
    process_all_markdown_files()