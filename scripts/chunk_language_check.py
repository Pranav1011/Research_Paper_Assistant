import os
import json
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42  # makes langdetect deterministic

INPUT_DIR = Path("data/enriched_chunks")
OUTPUT_DIR = Path("data/lang_enriched_chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def tag_languages():
    files = list(INPUT_DIR.glob("*_enriched.json"))
    if not files:
        print("❌ No enriched JSON files found in", INPUT_DIR)
        return

    for file_path in files:
        with open(file_path, "r") as f:
            chunks = json.load(f)

        for chunk in chunks:
            text = chunk.get("text", "")
            lang = detect_language(text)
            chunk["language"] = lang
            chunk["is_english"] = (lang == "en")
            chunk["length"] = len(text)

        out_path = OUTPUT_DIR / file_path.name.replace("_enriched.json", "_lang_enriched.json")
        with open(out_path, "w") as f:
            json.dump(chunks, f, indent=2)

        print(f"✅ Tagged: {out_path.name} ({len(chunks)} chunks)")

if __name__ == "__main__":
    tag_languages()