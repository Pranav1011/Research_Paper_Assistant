# Deep Researcher TS

A modern research assistant built with Next.js, TypeScript, and FastAPI. This project combines a TypeScript/React frontend with a Python FastAPI backend to perform real web searches and generate structured research summaries using local LLMs via Ollama.

## Features

- **Frontend**: Built with Next.js and TypeScript, featuring a modern UI with tabs for displaying research results.
- **Backend**: FastAPI-based Python server that performs real web searches using DuckDuckGo and processes results with local LLMs (Llama 2, Mistral, Llama 3, DeepSeek, Qwen2, etc.).
- **Structured Output**: Research results are organized into sections: Summary, Key Findings, Trends in Industry, Future Trends, Sources, and Process.
- **Dynamic Model Selection**: Choose from multiple local LLMs via a dropdown in the frontend.

## Prerequisites

- Node.js (v14 or later)
- Python (v3.8 or later)
- Ollama installed and running locally

## Setup

### Frontend

1. Navigate to the frontend directory:
   ```sh
   cd frontend
   ```

2. Install dependencies:
   ```sh
   npm install
   ```

3. Start the development server:
   ```sh
   npm run dev
   ```

### Backend

1. Navigate to the backend directory:
   ```sh
   cd backend
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```sh
   uvicorn main:app --reload
   ```

## Usage

1. Open your browser and navigate to `http://localhost:3000`.
2. Enter a research topic and select a model from the dropdown.
3. Submit the form to receive a structured research summary.

## Project Structure

- `frontend/`: Contains the Next.js/TypeScript frontend code.
- `backend/`: Contains the FastAPI backend code.
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
