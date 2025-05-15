import os
import arxiv
import requests
import PyPDF2
import io
from datetime import datetime
from typing import List, Dict
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivDownloader:
    def __init__(self, output_dir: str = "data/papers"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def search_papers(self, 
                     query: str,
                     max_results: int = 50,
                     sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
                     sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending) -> List[arxiv.Result]:
        """
        Search for papers on arXiv based on the given query.
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return list(search.results())

    def download_pdf(self, paper: arxiv.Result) -> str:
        """
        Download a paper's PDF and return the local path.
        """
        try:
            # Create a filename from the paper ID
            filename = f"{paper.entry_id.split('/')[-1]}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                logger.info(f"Paper {filename} already exists, skipping download")
                return filepath
            
            # Download the PDF
            response = requests.get(paper.pdf_url)
            response.raise_for_status()
            
            # Save the PDF
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading paper {paper.entry_id}: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None

    def process_paper(self, paper: arxiv.Result) -> Dict:
        """
        Process a single paper: download PDF and extract text.
        """
        # Download PDF
        pdf_path = self.download_pdf(paper)
        if not pdf_path:
            return None
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Create metadata
        metadata = {
            'id': paper.entry_id.split('/')[-1],
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'published': paper.published.strftime('%Y-%m-%d'),
            'summary': paper.summary,
            'pdf_path': pdf_path,
            'text': text
        }
        
        # Save metadata and text
        base_path = os.path.splitext(pdf_path)[0]
        with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        
        return metadata

    def download_papers(self, 
                       queries: List[str],
                       max_results_per_query: int = 50) -> List[Dict]:
        """
        Download papers for multiple queries.
        """
        all_metadata = []
        
        for query in queries:
            logger.info(f"Searching for papers with query: {query}")
            papers = self.search_papers(query, max_results=max_results_per_query)
            
            for paper in tqdm(papers, desc=f"Processing papers for query: {query}"):
                metadata = self.process_paper(paper)
                if metadata:
                    all_metadata.append(metadata)
        
        return all_metadata

def main():
    # Define search queries
    queries = [
        "large language models AND (retrieval OR search OR information retrieval)",
        "document retrieval AND (semantic search OR vector search)",
        "information retrieval AND (neural networks OR deep learning)",
        "text retrieval AND (embeddings OR vector space)",
        "document search AND (ranking OR relevance)"
    ]
    
    # Initialize downloader
    downloader = ArxivDownloader()
    
    # Download papers
    metadata = downloader.download_papers(queries, max_results_per_query=20)
    
    logger.info(f"Successfully downloaded and processed {len(metadata)} papers")

if __name__ == "__main__":
    main() 