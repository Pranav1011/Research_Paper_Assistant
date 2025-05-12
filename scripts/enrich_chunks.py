import os
import json
from pathlib import Path
import re

CHUNK_DIR = Path("data/chunks")
OUTPUT_DIR = Path("data/enriched_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_section_title(text):
    matches = re.findall(r"^#{2,3}\s+(.*)", text, re.MULTILINE)
    return matches[0].strip() if matches else "Unknown"

def enrich_chunks():
    files = list(CHUNK_DIR.glob("*_chunks.json"))
    if not files:
        print("❌ No chunk files found in data/chunks/")
        return

    for file_path in files:
        with open(file_path) as f:
            chunks = json.load(f)

        enriched = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            section_title = extract_section_title(text)
            enriched.append({
                "chunk_id": i,
                "text": text,
                "section_title": section_title,
                "source_file": file_path.name.replace("_chunks.json", ".md")
            })

        output_path = OUTPUT_DIR / file_path.name.replace("_chunks.json", "_enriched.json")
        with open(output_path, "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"✅ Saved enriched: {output_path.name} ({len(enriched)} chunks)")

if __name__ == "__main__":
    enrich_chunks()