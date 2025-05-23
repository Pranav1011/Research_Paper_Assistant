# Deep Researcher TS

A modern, full-stack research assistant for deep PDF and web research, powered by Google Gemini 2.0 Flash (free tier) and local LLMs (Llama 3, Qwen2) for web search augmentation. Built with FastAPI (Python) backend and Next.js/React frontend.

---

## Features

- **LLM-powered research:** Uses Gemini 2.0 Flash (free tier) for all core research and PDF analysis.
- **Web search augmentation:** Optionally uses Llama 3 or Qwen2 (via Ollama) to summarize and synthesize real-time web search results.
- **PDF support:** Upload a PDF and get a structured, detailed summary and insights.
- **Streaming-ready:** Frontend and backend are ready for streaming (word-by-word/chunked) answers.
- **Modern UI:** Chat-like, dark mode, input at the bottom, results above.
- **Minimal dependencies:** No paid LLMs required for core research.

---

## Model Usage & Architecture

- **Gemini 2.0 Flash** (Google, free tier):
  - Used for all core research, PDF analysis, and structured answers.
  - Chosen for its free tier, large context window, and web search grounding.
- **Llama 3 & Qwen2** (via Ollama):
  - Used for web search augmentation/fallback.
  - When web search is enabled, results from DuckDuckGo are summarized using Llama 3 or Qwen2.
  - No paid API keys required for these local models.

---

## Setup

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd deep-researcher-ts/backend
```

2. **Backend setup**
   - See [`backend/README.md`](README.md) for full backend instructions.
   - Requires Python, FastAPI, and a Google API key for Gemini 2.0 Flash.

```bash
pip install -r requirements.txt
```

3. **Set up your Google API key**

- Get a [Google AI Studio API key](https://aistudio.google.com/app/apikey)
- Create a `.env` file:

```
GOOGLE_API_KEY=your-google-api-key-here
```

4. **Run the backend**

```bash
uvicorn app.main:app --reload
```

5. **Run the Front end**

```bash
npm run dev
```

---

## API Overview

### `POST /api/research`
- **Description:** Analyze a research query and PDF using Gemini 2.0 Flash.
- **Form fields:**
  - `query` (str): Your research question/topic
  - `file` (PDF): PDF file to analyze (required)
- **Returns:**
  - `summary`: Structured, detailed answer
  - `sources`: List of sources (PDF, web, etc.)
  - `process`: Explanation of the research process

### `POST /api/websearch`
- **Description:** (Optional) Augment research with real-time web search (DuckDuckGo)
- **Body:**
  - `query` (str): Your research question/topic
- **Returns:**
  - `summary`: Web search summary
  - `sources`: List of web sources
  - `process`: Explanation of the process

- **/api/research**: Analyze a research query and PDF using Gemini 2.0 Flash.
- **/api/websearch**: Augment research with real-time web search, summarized by Llama 3 or Qwen2.
---

## Model Info

- **Model used:** `gemini-2.0-flash` (Google Gemini 2.0 Flash)
- **Why?**
  - Free tier, generous limits (1M context, web search grounding, etc.)
  - No paid tokens required
  - Reliable, fast, and supports structured prompting
- **No other LLMs are used.**

---

## Streaming Support

- The frontend is ready for streaming output (word-by-word/chunked answers).
- To enable streaming, add a `/api/research/stream` endpoint using FastAPI's `StreamingResponse` and Gemini's streaming API (if/when available).

---

## Project Decisions & Highlights

- **Frontend:** Next.js/React, beautiful dark mode, chat-like UX, streaming-ready.
- **Backend:** FastAPI, Gemini 2.0 Flash for research, Llama 3/Qwen2 for web search, robust error handling.
- **No paid LLMs:** Only free-tier Gemini 2.0 Flash and local models are used.
- **Attribution:** Developed by Sai Pranav.

---

## Contributing

PRs and issues welcome! Please open an issue for bugs, feature requests, or questions.

---

## License

MIT