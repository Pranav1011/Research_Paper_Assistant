import arxiv
import os

def download_papers(query="large language models", max_results=5, save_dir="data/papers"):
    os.makedirs(save_dir, exist_ok=True)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()
    
    for result in client.results(search):
        paper_id = result.get_short_id()
        safe_title = result.title.replace(" ", "_").replace("/", "_")[:80]  # truncate long titles
        filename = f"{paper_id}.{safe_title}.pdf"
        filepath = os.path.join(save_dir, filename)
        
        try:
            print(f" Downloading: {result.title}")
            result.download_pdf(dirpath=save_dir, filename=filename)
        except Exception as e:
            print(f"Failed to download {result.title}: {e}")
            continue

        # Validate it's a real PDF
        filetype = os.popen(f'file "{filepath}"').read()
        if "PDF document" not in filetype:
            print(f"Skipping non-PDF file: {filepath}")
            os.remove(filepath)

if __name__ == "__main__":
    download_papers()