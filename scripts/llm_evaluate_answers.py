import os
import sys
import json
import numpy as np
import random
import csv
import time
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: Please set the GEMINI_API_KEY environment variable.")
    sys.exit(1)

configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-flash")

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
embedder = SentenceTransformer(EMBED_MODEL_NAME)

TEST_QUERIES_PATH = 'data/paper_specific_test_queries.json'
CHUNKS_DIR = 'data/final_chunks'
OUTPUT_CSV = 'evaluation_results/llm_eval_results.csv'


def load_chunks(paper_id):
    chunk_file = Path(f"{CHUNKS_DIR}/{paper_id}_final_chunks.json")
    if not chunk_file.exists():
        print(f"Warning: Chunk file not found for paper_id {paper_id}")
        return []
    with open(chunk_file, 'r') as f:
        chunks = json.load(f)
    return chunks

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False)

def retrieve_top_k_chunks(chunks, question, k=5):
    if not chunks:
        return []
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

def build_judge_prompt(question, answer):
    return f"""Does the following answer correctly and fully address the question, based on the context?\nAnswer 'yes' or 'no' and provide a brief justification.\n\nQuestion: {question}\n\nAnswer: {answer}\n\nJudgement (yes/no) and justification:"""

def parse_judge_response(response_text):
    # Try to extract yes/no and justification
    lines = response_text.strip().split('\n')
    first_line = lines[0].strip().lower()
    if 'yes' in first_line:
        verdict = 'yes'
    elif 'no' in first_line:
        verdict = 'no'
    else:
        verdict = 'unknown'
    justification = response_text.strip()
    return verdict, justification

def main():
    # Optional: limit number of queries via command line argument
    max_queries = 10
    if len(sys.argv) > 1:
        try:
            max_queries = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for max_queries, using default 10.")
    # Load and sample queries
    with open(TEST_QUERIES_PATH, 'r') as f:
        all_queries = json.load(f)
    random.seed(42)
    sample_queries = random.sample(all_queries, min(len(all_queries)//2, max_queries))
    results = []
    for q in sample_queries:
        paper_id = q['paper_id']
        question = q['query']
        expected_section = q['expected_section']
        chunks = load_chunks(paper_id)
        if not chunks:
            continue
        top_chunks = retrieve_top_k_chunks(chunks, question, k=5)
        source_sections = "; ".join([chunk.get('section_title', 'Unknown') for chunk in top_chunks])
        prompt = build_prompt(top_chunks, question)
        answer = ask_gemini(prompt)
        time.sleep(5)  # Wait 5 seconds after answer generation
        judge_prompt = build_judge_prompt(question, answer)
        judge_response = ask_gemini(judge_prompt)
        time.sleep(5)  # Wait 5 seconds after judge call
        verdict, justification = parse_judge_response(judge_response)
        results.append({
            'query': question,
            'expected_section': expected_section,
            'answer': answer,
            'source_sections': source_sections,
            'judge_response': verdict,
            'judge_justification': justification
        })
    # Write to CSV
    os.makedirs('evaluation_results', exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['query', 'expected_section', 'answer', 'source_sections', 'judge_response', 'judge_justification'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Evaluation complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 