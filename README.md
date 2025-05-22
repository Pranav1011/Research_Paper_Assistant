# Deep Researcher TS – Backend (FastAPI + Gemini 2.0 Flash)

A modern research assistant backend powered by Google Gemini 2.0 Flash (free tier), with robust PDF analysis, real web search, and streaming-ready endpoints. Built for seamless integration with a Next.js/React frontend.

---

## Features

- **LLM-powered research**: Uses Gemini 2.0 Flash (free tier) for all research queries and PDF analysis.
- **PDF support**: Upload a PDF and get a structured, detailed summary and insights.
- **Web search integration**: Optionally augment research with real-time web search (DuckDuckGo, streaming-ready).
- **Streaming output**: Frontend is ready for streaming (word-by-word/chunked) answers.
- **Modern API**: FastAPI backend, CORS enabled, easy to extend.
- **Minimal dependencies**: No paid LLMs, no unnecessary bloat.

---

## Setup

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd deep-researcher-ts/backend
```

2. **Install dependencies**

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

---

## API Endpoints

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
- **Backend:** FastAPI, only Gemini 2.0 Flash, robust error handling, PDF parsing, and web search.
- **No paid LLMs:** Only free-tier Gemini 2.0 Flash is used for all research.
- **Modern UI:** Input at the bottom, results above, minimalist background, and compact forms.
- **Attribution:** Developed by Sai Pranav.

---

## Contributing

PRs and issues welcome! Please open an issue for bugs, feature requests, or questions.

---

## License

MIT (or your preferred license) 