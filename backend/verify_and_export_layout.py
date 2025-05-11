import json
from pathlib import Path
from backend.layout_parser import extract_layout_elements

EXPORT_DIR = Path("data/manual_parsed")

def sanitize_filename(name):
    return name.replace("?", "").replace(":", "").replace("/", "_")

def verify_and_export():
    papers_dir = Path("data/papers")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = list(papers_dir.glob("*.pdf"))
    print(f" Found {len(pdfs)} PDFs to validate")

    for pdf in pdfs:
        print(f"\n Verifying: {pdf.name}")
        try:
            parsed = extract_layout_elements(str(pdf))

            has_text = parsed["text"].strip() != ""
            has_tables = "[TABLE" in parsed["tables"]
            has_captions = "[CAPTION" in parsed["captions"]

            if not (has_text or has_tables or has_captions):
                print(" No useful content found, skipping.")
                continue

            # Save as JSON for Phase 4
            filename = sanitize_filename(pdf.stem) + ".json"
            with open(EXPORT_DIR / filename, "w") as f:
                json.dump(parsed, f, indent=2)
            print(f"Exported to: {EXPORT_DIR/filename}")

        except Exception as e:
            print(f"Error processing {pdf.name}: {e}")

if __name__ == "__main__":
    verify_and_export()