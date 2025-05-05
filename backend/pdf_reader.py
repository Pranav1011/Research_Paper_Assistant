from pathlib import Path
from backend.nougat_ocr import ocr_with_nougat
import fitz

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    
def extract_pdf(pdf_path: str) -> str:
    """
    Try extracting text; fallback to OCR if nothing is found.
    """
    text = extract_text_from_pdf(pdf_path)
    if len(text.strip()) < 500:  # heuristic
        print("Text too short — falling back to OCR...")
        text = ocr_with_nougat(pdf_path)
    return text