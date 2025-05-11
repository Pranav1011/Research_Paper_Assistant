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


## 🛠️ Setup
bash
# Clone repo
git clone https://github.com/yourname/research_paper_assistant
cd research_paper_assistant

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install MinerU
pip install -U "magic-pdf[full]"
pip install huggingface_hub
python download_models_hf.py

---

🧪 Run MinerU Parsing
magic-pdf -p data/papers/ -o data/mineru_parsed/

---

✅ Next Up: Phase 5 — Chunking + Embedding

Stay tuned! We’ll process the structured JSONs into semantic chunks for retrieval.

---


💻 Local-Only Focus

This app is designed for private use — no cloud APIs required.
