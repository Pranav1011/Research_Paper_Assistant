import gradio as gr
import os
from pathlib import Path
import subprocess
import tempfile
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf_and_answer(pdf_file, question):
    try:
        if pdf_file is None:
            return "Please upload a PDF file first."
        
        if not question:
            return "Please enter a question."
        
        # Get the original filename
        original_filename = os.path.basename(pdf_file.name)
        logger.info(f"Processing PDF: {original_filename}")
        logger.info(f"Question: {question}")
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to the temporary directory with a simple name
            temp_pdf_path = os.path.join(temp_dir, "input.pdf")
            shutil.copy2(pdf_file.name, temp_pdf_path)
            
            logger.info(f"Saved PDF to temporary path: {temp_pdf_path}")
            
            # Run the pipeline without evaluation
            cmd = ["python", "automated_pipeline.py", temp_pdf_path, question]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = f"Pipeline error: {result.stderr}"
                logger.error(error_msg)
                return error_msg
            
            logger.info("Pipeline completed successfully")
            logger.debug(f"Pipeline output: {result.stdout}")
            
            # Extract just the answer from the output
            output_lines = result.stdout.split('\n')
            answer = ""
            found_answer = False
            
            for line in output_lines:
                if "Answer:" in line:
                    found_answer = True
                    continue
                if found_answer and line.strip():
                    # Stop if we hit the next step marker or end of output
                    if "[1/4]" in line or "[2/4]" in line or "[3/4]" in line or "[4/4]" in line:
                        break
                    answer += line.strip() + "\n"
            
            if not answer:
                # If no answer was found in the pipeline output, try running llm_answer.py directly
                logger.info("No answer found in pipeline output, trying direct answer generation")
                cmd = ["python", "scripts/llm_answer.py", "input", question]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if "Answer:" in line:
                            found_answer = True
                            continue
                        if found_answer and line.strip():
                            answer += line.strip() + "\n"
            
            if not answer:
                logger.warning("No answer was generated")
                return "No answer was generated. Please try a different question."
            
            logger.info("Answer extracted successfully")
            return answer.strip()
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.exception(error_msg)
        return error_msg

# Create the Gradio interface
with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown("# 📚 Research Paper Assistant")
    gr.Markdown("Upload a research paper PDF and ask questions about its content.")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(
                label="Upload PDF", 
                file_types=[".pdf"],
                type="filepath"
            )
            question_input = gr.Textbox(
                label="Question", 
                placeholder="Enter your question about the paper...",
                lines=3
            )
            submit_btn = gr.Button("Get Answer", variant="primary")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer", 
                lines=10,
                show_copy_button=True
            )
    
    submit_btn.click(
        fn=process_pdf_and_answer,
        inputs=[pdf_input, question_input],
        outputs=answer_output,
        api_name="process_pdf"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True)