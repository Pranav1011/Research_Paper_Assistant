# Deep Researcher TS

A modern research assistant with a beautiful UI, built with Next.js (TypeScript) and FastAPI (Python). This project lets you upload PDFs, ask research questions, and get structured, AI-powered answers using local LLMs or Gemini (free tier), with robust fallback to web search.

## Features

- **Modern UI**: Next.js/TypeScript frontend with tabs, markdown rendering, and beautiful formatting.
- **PDF Upload**: Upload a PDF and ask research questions; the AI answers based on the document.
- **AI Models**: Uses Gemini (free tier) for document Q&A. If unavailable, falls back to web search + local LLMs (Ollama, Llama 3, etc.).
- **Web Search**: Optionally include real-time web search in your research.
- **Structured Output**: Results are organized into sections: Results, Key Findings, Trends in Industry, Future Trends, and Process.
- **Robust Error Handling**: Graceful fallback if any service is down or rate-limited.
- **ColiVara Ready**: ColiVara integration is preserved in the backend for future use.

## Prerequisites

- Node.js (v16 or later)
- Python (v3.9 or later)
- (Optional) Ollama installed and running locally for local LLM fallback
- (Optional) ColiVara API key for future advanced document search
- Google Gemini API key (free tier supported)

## Setup

### Frontend

1. Install dependencies:
   ```sh
   npm install
   ```
2. Start the development server:
   ```sh
   npm run dev
   ```

### Backend

1. Install dependencies:
   ```sh
   cd backend
   pip install -r requirements.txt
   ```
2. Add your Gemini API key to `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
3. Start the FastAPI server:
   ```sh
   uvicorn app.main:app --reload
   ```

## Usage

1. Open your browser and go to `http://localhost:3000`.
2. Enter a research topic, upload a PDF, and (optionally) enable web search.
3. Click "Start Research" to get a structured, AI-powered answer.
4. Results are beautifully formatted, with bullet points and sections.

## Project Structure

- `src/` — Next.js frontend (components, pages, styles, etc.)
- `backend/app/main.py` — FastAPI backend (all logic here)
- `backend/requirements.txt` — Backend dependencies
- `README.md` — Project documentation

### File/Folder Cleanup Recommendations
- `backend/app/test.py` — Old ColiVara test script. Delete or move to `backend/experiments/` if you want to keep it for reference.
- `src/types/` — Currently empty. Delete if not planning to use.
- `scripts/setup.sh` — Delete if not used.
- `components.json` — Delete if not used by your build/deployment.
- `.next/`, `node_modules/`, `venv/` — Should be in `.gitignore` (not committed).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License.

