# Deep Researcher Python Backend

This is the Python backend for the Deep Researcher project, built with FastAPI and LangChain.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the server:
```bash
uvicorn app.main:app --reload
```

The server will start at http://localhost:8000

## API Endpoints

### POST /api/research
Research endpoint that processes queries using LangChain and OpenAI.

Request body:
```json
{
    "query": "Your research query",
    "context": "Optional context"
}
```

Response:
```json
{
    "result": "Research results",
    "sources": ["List of sources"]
}
```

## Development

- The main application is in `app/main.py`
- Uses FastAPI for the web framework
- LangChain for AI/ML processing
- OpenAI GPT-4 for the language model 