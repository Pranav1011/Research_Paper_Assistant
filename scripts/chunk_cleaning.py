import os
import json
from pathlib import Path
import re
from collections import defaultdict

INPUT_DIR = Path("data/lang_enriched_chunks")
OUTPUT_DIR = Path("data/final_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def is_symbol_heavy(text, threshold=0.6):
    if not text.strip():
        return True
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return (symbols / max(len(text), 1)) > threshold

def clean_chunks():
    files = list(INPUT_DIR.glob("*.json"))
    if not files:
        print("❌ No enriched chunk files found.")
        return

    for path in files:
        with open(path) as f:
            chunks = json.load(f)

        seen = set()
        cleaned = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()

            if text == "":
                continue
            if len(text.split()) < 5:
                continue
            if is_symbol_heavy(text):
                continue
            if text in seen:
                continue  # optional: skip duplicate chunks

            seen.add(text)
            cleaned.append(chunk)

        filename = path.name.replace("_lang_enriched.json", "_final.json")
        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(cleaned, f, indent=2)

        print(f"✅ Cleaned: {filename} ({len(cleaned)} chunks retained)")

if __name__ == "__main__":
    clean_chunks()