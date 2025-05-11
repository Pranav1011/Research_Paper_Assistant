# 🧠 Research Paper Assistant (with MinerU)

A local RAG pipeline for parsing, indexing, and querying academic PDFs — powered by LLMs, FAISS, and MinerU.

---

## ✅ Completed Phases

### 1. Project Setup
- GitHub repo structured
- Python virtual environment

### 2. ArXiv Scraper
- Downloads latest LLM papers using `arxiv` Python package
- Stores to `data/papers/`

### 3. Manual PDF Parsing (Deprecated)
- Used `PyMuPDF` and `Nougat OCR` for extracting text and layout
- Outputs to `data/manual_parsed/`

### 4. MinerU Setup
- Installed via `magic-pdf[full]`
- Downloaded model weights from HuggingFace
- Parsed PDFs into layout-aware JSON and Markdown
- Output format: `data/mineru_parsed/{filename}/model.json`

---

## 🗂️ File Structure