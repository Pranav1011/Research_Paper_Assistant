# Research Paper Assistant

A tool that helps you understand research papers by answering questions about their content. The assistant processes PDF papers and uses advanced language models to provide accurate answers to your questions.

## Features

- PDF parsing and text extraction
- Intelligent text chunking and processing
- Question answering using advanced language models
- User-friendly web interface
- Detailed logging for debugging

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research_paper_assistant.git
cd research_paper_assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the web interface:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:7860`

3. Upload a PDF research paper and ask questions about its content

### Command Line Interface

You can also use the tool from the command line:

```bash
python automated_pipeline.py <pdf_path> [question] [--evaluate]
```

Options:
- `<pdf_path>`: Path to the PDF file (required)
- `[question]`: Question about the paper (optional)
- `[--evaluate]`: Run evaluation steps (optional)

## Project Structure

```
research_paper_assistant/
├── app.py                 # Web interface
├── automated_pipeline.py  # Main processing pipeline
├── scripts/              # Processing scripts
│   ├── chunking_md_sections.py
│   ├── process_chunks.py
│   └── llm_answer.py
├── data/                 # Data directory
│   ├── chunks/          # Temporary chunk files
│   ├── final_chunks/    # Processed chunks
│   └── mineru_parsed/   # PDF parsing output
├── tests/               # Test files
└── requirements.txt     # Python dependencies
```

## Development

### Running Tests

The project includes several test files for different components:

- `tests/eval_section_matching.py`: Tests for section matching
- `tests/test_hybrid_retriever.py`: Tests for hybrid retrieval
- `tests/query_faiss_index.py`: Tests for FAISS index querying
- `tests/validate_mineru_output.py`: Tests for PDF parser output validation

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
