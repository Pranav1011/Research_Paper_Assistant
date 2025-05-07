import pdfplumber
import re

def extract_layout_elements(pdf_path: str):
    all_text = []
    all_tables = []
    all_captions = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            #  Text with column handling
            text = page.extract_text(layout=True, vertical_strategy="lines", dedupe_chars=True)
            if text:
                all_text.append(f"[PAGE {i+1}]\n{text}")

            # Extract tables
            tables = page.extract_tables()
            lines = text.split("\n") if text else []

            for table in tables:
                formatted_table = "\n".join([
                    "\t".join(cell if cell is not None else "" for cell in row)
                    for row in table if row
                ])
                
                #  Find nearby caption for this table
                caption_match = None
                for line in lines:
                    if re.search(r"(Table|Figure)\s?\d+[:.]?", line, re.IGNORECASE):
                        caption_match = line.strip()
                        break

                if caption_match:
                    formatted_table = f"{caption_match}\n{formatted_table}"
                    all_captions.append(f"[CAPTION Page {i+1}]\n{caption_match}")

                all_tables.append(f"[TABLE Page {i+1}]\n{formatted_table}")

    return {
        "text": "\n\n".join(all_text),
        "tables": "\n\n".join(all_tables),
        "captions": "\n\n".join(all_captions)
    }