# 🧠 Research Paper Assistant (with MinerU)

A local RAG pipeline for parsing, indexing, and querying academic PDFs — powered by LLMs, FAISS, and MinerU.

---

## ✅ Completed Phases

### 1. Project Setup

* GitHub repo structured
* Python virtual environment initialized

### 2. ArXiv Scraper

* Downloads latest LLM papers using `arxiv` Python package
* Stores to `data/papers/`

### 3. Manual PDF Parsing (Deprecated)

* Used `PyMuPDF` and `Nougat OCR` for extracting text and layout
* Output to `data/manual_parsed/` (now removed)

### 4. MinerU Setup

* Installed via `magic-pdf[full]`
* Downloaded model weights from HuggingFace
* Parsed PDFs into layout-aware JSON and Markdown
* Output format: `data/mineru_parsed/{filename}/model.json`

---

## 📅 Phase 5: Chunking + Embedding

### ✨ Goal

Transform parsed academic papers (from MinerU) into semantic chunks and generate embeddings for efficient and meaningful retrieval.

### ✅ What Was Done

#### ✏️ Chunk Processing

* **Script:** `scripts/process_chunks.py`
* Reads from: `data/mineru_parsed/`
* Actions:

  * Cleans, deduplicates, and filters raw Markdown content.
  * Splits text into semantically coherent chunks.
  * Matches each chunk to the closest section title based on embedding similarity.
  * Adds metadata: section title, chunk ID, source file, language, etc.
* Outputs to: `data/final_chunks/`

#### 🤖 Embedding

* **Model used:** `sentence-transformers/all-MiniLM-L6-v2`
* **Actions:**

  * Embeds each section title and chunk for semantic matching and retrieval.
  * Stores chunk embeddings for indexing in FAISS.

#### 📘 Section Matching

* Extracts headers from markdown.
* Computes similarity between chunk embedding and section embeddings.
* Assigns closest semantic match with optional similarity threshold.

#### ⚖️ Reproducibility

* All steps automated via `process_chunks.py`.
* Deterministic and reproducible outputs.

---

## 🤖 Phase 6: Hybrid Retrieval Engine (FAISS + BM25)

### ✨ Goal

Enable robust, semantically-enriched retrieval using both dense (FAISS) and sparse (BM25) methods.

### ✅ What Was Done

#### 🤝 Hybrid Retriever

* **File:** `vectorstore/hybrid_retriever.py`
* **Components:**

  * **FAISS Index**: fast dense retrieval using chunk embeddings.
  * **BM25 Index**: keyword-based scoring using rank\_bm25.
  * **Section Boosting**: boosts results with section-title matches.
  * **Query Expansion**: optional synonym expansion via NLTK.
  * **Reranker**: optional cross-encoder reranking for top-k results.
  * **Score Fusion**: customizable weights for FAISS, BM25, and reranker.

#### 🔍 Testing & CLI

* **File:** `tests/test_hybrid_retriever.py`
* Includes:

  * Retrieval testing and debugging.
  * Score analysis and section match evaluation.
  * CLI options for filtering, reranking, debugging scores.

#### 📂 Data Cleanup

* Deprecated manual parsing (`data/manual_parsed/`) removed.
* Now relies solely on MinerU + chunking + embeddings.

#### ♻️ Extensible

* Modular design: easy to add new retrieval methods, scoring strategies, or filtering logic.

---

## ⚡ What's Next

### 📁 Phase 7: LLM Routing Layer + FastAPI Backend (Upcoming)

* Expose the hybrid retriever as a `/ask` endpoint.
* Use in downstream apps or frontend interface.

---

## 🗂️ File/Directory Summary

| File/Folder                       | Purpose                                               |
| --------------------------------- | ----------------------------------------------------- |
| `scripts/process_chunks.py`       | Chunking, cleaning, section assignment, and embedding |
| `data/mineru_parsed/`             | MinerU-parsed JSON + markdown                         |
| `data/final_chunks/`              | Final enriched and cleaned chunks                     |
| `vectorstore/hybrid_retriever.py` | Core logic for hybrid retrieval                       |
| `tests/test_hybrid_retriever.py`  | CLI tool for evaluating and debugging retrieval       |

---

## 🚀 Run MinerU to Parse PDFs

```bash
magic-pdf -p data/papers/ -o data/mineru_parsed/
```

## 🎓 Run Chunking + Embedding

```bash
python scripts/process_chunks.py
```

## 🔬 Run Retrieval Test

```bash
python tests/test_hybrid_retriever.py
```
