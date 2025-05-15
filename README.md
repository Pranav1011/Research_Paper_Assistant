# 🧠 Research Paper Assistant (with MinerU)

A local RAG pipeline for parsing, indexing, querying, and evaluating academic PDFs — powered by LLMs, FAISS, and MinerU.

---

## ✅ Completed Phases

### 1. Project Setup

* GitHub repo structured
* Python virtual environment initialized

### 2. ArXiv Scraper

* Downloads latest LLM papers using `arxiv` Python package
* Stores to `data/papers/`

### 3. MinerU Setup

* Installed via `magic-pdf[full]`
* Downloaded model weights from HuggingFace
* Parsed PDFs into layout-aware JSON and Markdown
* Output format: `data/mineru_parsed/{filename}/model.json`

### 4. Chunking + Embedding

* **Script:** `scripts/process_chunks.py`
* Cleans, deduplicates, and filters raw Markdown content
* Splits text into semantically coherent chunks
* Matches each chunk to the closest section title based on embedding similarity
* Adds metadata: section title, chunk ID, source file, language, etc.
* Outputs to: `data/final_chunks/`
* Embeds each chunk and section title for semantic retrieval (FAISS)

### 5. Hybrid Retrieval Engine (FAISS + BM25)

* **File:** `vectorstore/hybrid_retriever.py`
* Combines dense (FAISS) and sparse (BM25) retrieval
* Section boosting, query expansion, reranking, and score fusion
* CLI and test tools in `tests/`

### 6. LLM Routing, Answer Generation, and Evaluation (Phase 7 — Complete!)

* **Scripts:**
  * `scripts/llm_answer.py`: LLM-based answer generation from top-k retrieved chunks
  * `scripts/llm_evaluate_answers.py`: Automated LLM-based answer evaluation pipeline
  * `scripts/llm_eval_metric.py`: Computes accuracy from LLM evaluation results
* **Pipeline:**
  * PDF upload → MinerU parsing → Chunking/embedding → Retrieval → LLM answer → LLM evaluation
  * All steps can be run via CLI scripts
* **Backend:**
  * `main.py` provides a FastAPI entry point (for future API integration)

---

## 🗂️ File/Directory Summary

| File/Folder                       | Purpose                                               |
| --------------------------------- | ----------------------------------------------------- |
| `scripts/arxiv_downloader.py`     | Download LLM papers from arXiv                        |
| `scripts/process_chunks.py`       | Chunking, cleaning, section assignment, and embedding |
| `data/mineru_parsed/`             | MinerU-parsed JSON + markdown                         |
| `data/final_chunks/`              | Final enriched and cleaned chunks                     |
| `vectorstore/hybrid_retriever.py` | Core logic for hybrid retrieval                       |
| `scripts/llm_answer.py`           | LLM answer generation from top-k chunks               |
| `scripts/llm_evaluate_answers.py` | LLM-based answer evaluation pipeline                  |
| `scripts/llm_eval_metric.py`      | Computes accuracy from LLM evaluation results         |
| `evaluation_results/`             | LLM evaluation results (CSV)                          |
| `main.py`                         | FastAPI entry point                                   |

---

## 🚀 Usage

### 1. Download Papers from ArXiv
```bash
python scripts/arxiv_downloader.py
```

### 2. Parse PDFs with MinerU
```bash
magic-pdf -p data/papers/ -o data/mineru_parsed/
```

### 3. Chunk, Clean, and Embed
```bash
python scripts/process_chunks.py
```

### 4. Generate LLM Answers (Gemini API)
```bash
python scripts/llm_answer.py <paper_id> <question>
```

### 5. Evaluate LLM Answers (Automated)
```bash
python scripts/llm_evaluate_answers.py [max_queries]
```
- Results saved to `evaluation_results/llm_eval_results.csv`

### 6. Compute LLM Evaluation Metric
```bash
python scripts/llm_eval_metric.py
```

---

## ⚙️ Environment Variables
- Set your Gemini API key in a `.env` file or environment variable:
  ```
  GEMINI_API_KEY=your_gemini_api_key_here
  ```

---

## 📝 Notes
- All steps are fully automated and reproducible via CLI scripts.
- The pipeline is modular: you can swap out chunking, retrieval, or LLM components as needed.
- For API integration, extend `main.py` and the `backend/` directory.
- For new PDFs, simply repeat steps 1–6.

---

## 🎉 Phase 7 Complete!
- Full pipeline: PDF upload → MinerU parsing → Chunking/embedding → Retrieval → LLM answer → LLM evaluation
- Ready for research, benchmarking, and further extension!
