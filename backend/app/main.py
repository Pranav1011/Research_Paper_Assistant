from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import requests
import asyncio
import concurrent.futures
import traceback
import time
import random
import json
import re

# === More local LLMs you can use with Ollama/LMStudio (no API needed) ===
# - llama2, llama3, mistral, phi3, deepseek-llm, deepseek-coder, qwen1.5, qwen2, gemma, codellama, yi, solar, openhermes, neural-chat, etc.
# See https://ollama.com/library for more.

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str
    context: Optional[str] = None
    model: Optional[str] = "llama3"  # Default to llama3

class ResearchResponse(BaseModel):
    summary: str
    sources: List[dict]
    process: str = ""


def duckduckgo_search_sync(query: str, max_results: int = 3) -> list:
    results = []
    try:
        # Add a small random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # Configure DDGS with browser-like headers
        with DDGS() as ddgs:
            # Set custom headers to appear more like a regular browser
            ddgs.headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            # Use a more conservative search approach
            for r in ddgs.text(
                query,
                region="wt-wt",
                safesearch="Moderate",
                max_results=max_results,
                backend="lite"  # Use lite backend which is less likely to trigger rate limits
            ):
                results.append({
                    "title": r.get("title"),
                    "body": r.get("body"),
                    "href": r.get("href")
                })
                
                # Add a small delay between results
                time.sleep(random.uniform(0.5, 1))
                
    except DuckDuckGoSearchException as e:
        print(f"[DuckDuckGo] Search error: {e}")
        # If we hit a rate limit, wait longer and try one more time
        if "rate limit" in str(e).lower():
            print("Rate limit detected, waiting 30 seconds before retry...")
            time.sleep(30)
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(
                        query,
                        region="wt-wt",
                        safesearch="Moderate",
                        max_results=max_results,
                        backend="lite"
                    ):
                        results.append({
                            "title": r.get("title"),
                            "body": r.get("body"),
                            "href": r.get("href")
                        })
            except Exception as retry_e:
                print(f"[DuckDuckGo] Retry search error: {retry_e}")
                return []
        return []
    except Exception as e:
        print(f"[DuckDuckGo] Unexpected error: {e}")
        return []
    return results

def extract_sections(text):
    def get_section(name, next_names):
        next_pattern = "|".join([re.escape(n) for n in next_names])
        pattern = rf"{name}:\s*(.*?)(?:{next_pattern}:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    summary = get_section("Summary", ["Key Findings", "Trends in Industry", "Future Trends", "Process", "Sources"])
    key_findings = get_section("Key Findings", ["Trends in Industry", "Future Trends", "Process", "Sources"])
    trends = get_section("Trends in Industry", ["Future Trends", "Process", "Sources"])
    future = get_section("Future Trends", ["Process", "Sources"])
    process = get_section("Process", ["Sources"])
    sources = get_section("Sources", [])
    # Compose the summary with sub-sections
    summary_full = summary
    if key_findings:
        summary_full += f"\n\nKey Findings:\n{key_findings}"
    if trends:
        summary_full += f"\n\nTrends in Industry:\n{trends}"
    if future:
        summary_full += f"\n\nFuture Trends:\n{future}"
    return summary_full.strip(), process.strip(), sources.strip()

@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    try:
        # Dynamically create the Ollama model instance based on the requested model
        model = ChatOllama(
            base_url="http://localhost:11434",
            model=request.model,
            temperature=0.7
        )

        # Retrieve web context using a separate process to avoid async issues
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            web_results = await loop.run_in_executor(pool, duckduckgo_search_sync, request.query)
        web_context = "\n".join([f"{r['title']}: {r['body']} ({r['href']})" for r in web_results])

        print("QUERY:", request.query)
        print("WEB CONTEXT:", web_context)

        # Use a single LLM call for summary with embedded sub-sections
        prompt_str = (
            "You are a helpful research assistant. Using the web search context below, write a detailed, multi-paragraph summary (at least 8 sentences) answering the research question. "
            "Within your summary, include the following clearly marked sub-sections:\n"
            "- Key Findings: (as a short paragraph or bullet points)\n"
            "- Trends in Industry: (as a short paragraph or bullet points)\n"
            "- Future Trends: (as a short paragraph or bullet points)\n"
            "After the summary, provide a 'Process' section explaining step-by-step how you used the context to generate your answer.\n"
            "Format:\n"
            "Summary:\n[Your detailed summary here]\n"
            "Key Findings:\n[Bullets or paragraph]\n"
            "Trends in Industry:\n[Bullets or paragraph]\n"
            "Future Trends:\n[Bullets or paragraph]\n"
            "Process:\n[Step-by-step explanation]\n"
            "Sources:\n[List the main sources you used, with title and URL]\n"
            "Do not include any text before or after these sections.\n"
            f"Web Search Context:\n{web_context}\n\n"
            f"Research Question:\n{request.query}\n"
        )
        print("DEBUG FINAL PROMPT SENT TO LLM:\n", prompt_str)
        result = await model.ainvoke(prompt_str)
        result_text = result.content if hasattr(result, "content") else str(result)
        summary, process, sources_text = extract_sections(result_text)
        # Parse sources as a list of dicts if possible, else fallback to web_results
        parsed_sources = []
        # Try to extract sources as title/url pairs from the sources_text
        if sources_text:
            for line in sources_text.splitlines():
                m = re.match(r"-?\s*(.+?)\s*\((https?://[^)]+)\)", line)
                if m:
                    parsed_sources.append({"title": m.group(1).strip(), "href": m.group(2).strip(), "body": ""})
        # Fallback to web_results if no sources were parsed
        if not parsed_sources:
            parsed_sources = web_results
        return ResearchResponse(
            summary=summary,
            sources=parsed_sources,
            process=process
        )
    except Exception as e:
        print("Exception in /api/research:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 