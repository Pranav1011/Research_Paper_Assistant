import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Read Gemini API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: Please set the GEMINI_API_KEY environment variable.")
    sys.exit(1)

configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBED_MODEL_NAME)


def load_chunks(paper_id):
    chunk_file = Path(f"data/final_chunks/{paper_id}_final_chunks.json")
    if not chunk_file.exists():
        print(f"Error: Chunk file not found for paper_id {paper_id}")
        sys.exit(1)
    with open(chunk_file, 'r') as f:
        chunks = json.load(f)
    return chunks

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False)

def retrieve_top_k_chunks(chunks, question, k=5):
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = embed_texts(chunk_texts)
    question_embedding = embed_texts([question])[0]
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_chunks = [chunks[i] for i in top_k_indices]
    return top_chunks

def build_prompt(chunks, question):
    context = "\n\n".join([chunk['text'] for chunk in chunks])
    prompt = f"""You are a research assistant. Given the following context from a research paper and a user question, answer the question as accurately as possible using only the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    return prompt

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/llm_answer.py <paper_id> <question>")
        sys.exit(1)
    paper_id = sys.argv[1]
    question = " ".join(sys.argv[2:])
    print(f"\nPaper ID: {paper_id}\nQuestion: {question}\n")
    chunks = load_chunks(paper_id)
    top_chunks = retrieve_top_k_chunks(chunks, question, k=5)
    prompt = build_prompt(top_chunks, question)
    answer = ask_gemini(prompt)
    print("\n--- LLM Answer ---\n")
    print(answer)
    print("\n--- Source Sections Used ---\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Section {i}: {chunk.get('section_title', 'Unknown')}")

if __name__ == "__main__":
    main() 