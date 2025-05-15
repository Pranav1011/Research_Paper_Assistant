import json
import os
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read Gemini API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: Please set the GEMINI_API_KEY environment variable.")
    sys.exit(1)

configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-flash")  # Using flash for faster responses

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def get_relevant_chunks(pdf_name: str, question: str, top_k: int = 3):
    """
    Get the most relevant chunks for a question from the processed chunks file.
    
    Args:
        pdf_name: Name of the PDF file
        question: Question to find relevant chunks for
        top_k: Number of top chunks to return
        
    Returns:
        List of relevant chunks
    """
    # Load the processed chunks
    chunks_file = Path("data/final_chunks") / f"{pdf_name}_final_chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunk file not found for PDF: {pdf_name}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Get embeddings for chunks
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=False)
    
    # Get embedding for question
    question_embedding = embedder.encode([question], show_progress_bar=False)[0]
    
    # Calculate similarities
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )
    
    # Get top k chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices]

def generate_answer(question: str, chunks: list) -> str:
    """
    Generate an answer using Gemini based on the question and relevant chunks.
    
    Args:
        question: Question to answer
        chunks: List of relevant chunks
        
    Returns:
        Generated answer
    """
    # Prepare context from chunks
    context = "\n\n".join([
        f"Section: {chunk['section_title']}\n{chunk['text']}"
        for chunk in chunks
    ])
    
    # Create the prompt
    prompt = f"""Based on the following context from a research paper, please answer the question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer using Gemini
    response = model.generate_content(prompt)
    return response.text

def main():
    if len(sys.argv) < 3:
        print("Usage: python llm_answer.py <pdf_name> <question>")
        sys.exit(1)
    
    pdf_name = sys.argv[1]
    question = " ".join(sys.argv[2:])
    
    try:
        # Get relevant chunks
        chunks = get_relevant_chunks(pdf_name, question)
        
        # Generate answer
        answer = generate_answer(question, chunks)
        
        # Print answer
        print("\nAnswer:")
        print(answer)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 