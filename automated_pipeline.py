import os
import sys
import subprocess
from pathlib import Path
import glob

def clean_old_chunks(pdf_name):
    # Remove all files in data/chunks and data/final_chunks except for the current pdf_name
    for folder in ["data/chunks", "data/final_chunks"]:
        for f in glob.glob(f"{folder}/*"):
            if pdf_name not in f:
                os.remove(f)

def run_magic_pdf(pdf_path, output_dir):
    # Run Magic-PDF (mineru) via command line
    print(f"[1/4] Parsing PDF with Magic-PDF: {pdf_path}")
    result = subprocess.run([
        "magic-pdf", "--path", pdf_path, "--output-dir", str(output_dir)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running Magic-PDF: {result.stderr}")
        sys.exit(1)
    print("Magic-PDF parsing complete.")

def run_chunking_md_sections(pdf_name):
    print("[2/4] Chunking markdown sections...")
    result = subprocess.run(["python", "scripts/chunking_md_sections.py", pdf_name])
    if result.returncode != 0:
        print("Error running chunking_md_sections.py")
        sys.exit(1)
    print("Markdown sections chunked.")

def run_process_chunks(pdf_name):
    print("[3/4] Cleaning and merging chunks...")
    result = subprocess.run(["python", "scripts/process_chunks.py", pdf_name])
    if result.returncode != 0:
        print("Error running process_chunks.py")
        sys.exit(1)
    print("Chunks cleaned and merged.")

def run_llm_answer(pdf_name, question):
    print("[4/4] Generating answer with LLM...")
    result = subprocess.run(
        ["python", "scripts/llm_answer.py", pdf_name, question],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error running llm_answer.py: {result.stderr}")
        print(f"Stdout: {result.stdout}")
        sys.exit(1)
    print(result.stdout)
    return result.stdout

def run_llm_evaluate_answers(num_queries=1):
    print("[5/6] Evaluating answer with LLM judge...")
    result = subprocess.run(["python", "scripts/llm_evaluate_answers.py", str(num_queries)])
    if result.returncode != 0:
        print("Error running llm_evaluate_answers.py")
        sys.exit(1)
    print("LLM evaluation complete.")

def run_llm_eval_metric():
    print("[6/6] Computing accuracy metric...")
    result = subprocess.run(["python", "scripts/llm_eval_metric.py"])
    if result.returncode != 0:
        print("Error running llm_eval_metric.py")
        sys.exit(1)
    print("Metric computed.")

def get_pdf_name(pdf_path):
    """
    Extract the PDF filename without extension.
    """
    return Path(pdf_path).stem

def main(pdf_path, question=None, evaluate=False):
    # Get PDF name from the file path
    pdf_name = get_pdf_name(pdf_path)
    
    mineru_output_dir = Path("data/mineru_parsed")
    
    # Clean up old chunks except for the current pdf_name
    clean_old_chunks(pdf_name)
    
    # 1. Parse PDF with Magic-PDF
    run_magic_pdf(pdf_path, mineru_output_dir)
    
    # 2. Chunk markdown sections
    run_chunking_md_sections(pdf_name)
    
    # 3. Clean/merge chunks
    run_process_chunks(pdf_name)
    
    # 4. Get answer if question is provided
    if question:
        answer = run_llm_answer(pdf_name, question)
        
        # Only run evaluation steps if explicitly requested
        if evaluate:
            # 5. Evaluate answer
            run_llm_evaluate_answers(1)
            # 6. Print accuracy metric
            run_llm_eval_metric()
        
        return answer
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python automated_pipeline.py <pdf_path> [question] [--evaluate]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    question = " ".join(sys.argv[2:-1]) if len(sys.argv) > 2 and sys.argv[-1] != "--evaluate" else " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
    evaluate = "--evaluate" in sys.argv
    
    main(pdf_path, question, evaluate) 