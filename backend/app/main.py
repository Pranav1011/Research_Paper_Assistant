from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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
import base64
import google.generativeai as genai
from PyPDF2 import PdfReader
import io

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

# Initialize Gemini and list available models
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
available_models = [m.name for m in genai.list_models()]
print("Available Gemini models:", available_models)

# Use the first available model that can generate content
model = None
for model_name in available_models:
    if "generateContent" in genai.get_model(model_name).supported_generation_methods:
        model = genai.GenerativeModel(model_name)
        print(f"Using model: {model_name}")
        break

if model is None:
    print("Warning: No suitable Gemini model found. Will fall back to web search.")

# Preserve ColiVara code as comments for future use
"""
from colivara_py import ColiVara
colivara_client = ColiVara(api_key=os.getenv("COLIVARA_API_KEY"))
"""

class ResearchRequest(BaseModel):
    query: str
    context: Optional[str] = None
    model: Optional[str] = "llama3"  # Default to llama3

class ResearchResponse(BaseModel):
    summary: str
    sources: List[dict]
    process: str = ""

@app.post("/api/research", response_model=ResearchResponse)
async def research(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="PDF file is required for research.")
        
        # Save the uploaded file
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(await file.read())

        # Read PDF content
        pdf_reader = PdfReader(file_location)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Use Gemini to analyze the PDF
        prompt = f"""Please analyze the following PDF content and answer the query: {query}

PDF Content:
{pdf_text}

Please provide:
1. A comprehensive summary of the relevant information
2. Key findings and insights
3. Any important quotes or references
4. Page numbers where the information was found

Format your response in a clear, structured way."""

        try:
            if model is None:
                raise Exception("No Gemini model available")
            response = model.generate_content(prompt)
            summary = response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg and "quota" in error_msg.lower():
                # If we hit rate limits, fall back to web search
                print("Gemini rate limit hit, falling back to web search...")
                web_results = duckduckgo_search_sync(query)
                context = "\n".join([f"{r['title']}: {r['body']} ({r['href']})" for r in web_results])
                llm = ChatOllama(
                    base_url="http://localhost:11434",
                    model="llama3",
                    temperature=0.7
                )
                prompt = create_prompt(query, context)
                result = await llm.ainvoke(prompt)
                summary = result.content if hasattr(result, "content") else str(result)
            else:
                print(f"Error with Gemini: {error_msg}")
                print("Falling back to web search...")
                web_results = duckduckgo_search_sync(query)
                context = "\n".join([f"{r['title']}: {r['body']} ({r['href']})" for r in web_results])
                llm = ChatOllama(
                    base_url="http://localhost:11434",
                    model="llama3",
                    temperature=0.7
                )
                prompt = create_prompt(query, context)
                result = await llm.ainvoke(prompt)
                summary = result.content if hasattr(result, "content") else str(result)
        
        # Process the response
        sources = [{
            "title": file.filename,
            "href": "",
            "body": pdf_text[:500] + "...",  # First 500 chars as preview
            "page_image": "",
            "page_number": "1"  # We don't have page numbers from Gemini
        }]

        # Clean up
        try:
            os.remove(file_location)
        except Exception as e:
            print(f"Warning: could not delete temp file {file_location}: {e}")

        return ResearchResponse(
            summary=summary,
            sources=sources,
            process="Research conducted using Google's Gemini AI model."
        )

    except Exception as e:
        print(f"Error in research endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Preserve ColiVara implementation as comments
"""
@app.post("/api/research", response_model=ResearchResponse)
async def research(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="PDF file is required for ColiVara research.")
        # Save the uploaded file
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as file_object:
            file_object.write(await file.read())

        # Now open and read for base64 encoding
        with open(file_location, "rb") as f:
            file_content = f.read()
            encoded_content = base64.b64encode(file_content).decode("utf-8")

        document = colivara_client.upsert_document(
            name=file.filename,
            document_base64=encoded_content,
            collection_name="default_collection",
            wait=True
        )
        try:
            search_response = colivara_client.search(
                query,
                collection_name="default_collection",
                top_k=5
            )
            print("Raw search response:", search_response)
            results = getattr(search_response, "results", None)
            if results is None:
                raise Exception("ColiVara search did not return a .results attribute")
            print("Results type:", type(results))
            if results:
                print("First result type:", type(results[0]))
                print("First result dir:", dir(results[0]))
                print("First result dict:", getattr(results[0], '__dict__', None))
                print("First result repr:", repr(results[0]))
            context = "\n".join([getattr(r, "text", "") for r in results])
            sources = []
            for r in results:
                source = {
                    "title": getattr(r, "title", ""),
                    "href": getattr(r, "url", ""),
                    "body": getattr(r, "text", ""),
                    "page_image": getattr(r, "img_base64", ""),
                    "page_number": getattr(r, "page_number", "")
                }
                sources.append(source)
        except Exception as e:
            print(f"ColiVara error: {e}")
            raise HTTPException(status_code=502, detail="ColiVara server error. Please try again later or contact support.")
        # Only delete after all operations are done
        try:
            os.remove(file_location)
        except Exception as e:
            print(f"Warning: could not delete temp file {file_location}: {e}")
        return ResearchResponse(
            summary=context,
            sources=sources,
            process="Research conducted using ColiVara's advanced document analysis capabilities."
        )
    except Exception as e:
        print(f"Error in research endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""

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

@app.post("/api/websearch")
async def websearch(payload: dict):
    query = payload.get("query")
    model = payload.get("model", "llama3")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model=model,
        temperature=0.7
    )
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        web_results = await loop.run_in_executor(pool, duckduckgo_search_sync, query)
    context = "\n".join([f"{r['title']}: {r['body']} ({r['href']})" for r in web_results])
    prompt = create_prompt(query, context)
    result = await llm.ainvoke(prompt)
    result_text = result.content if hasattr(result, "content") else str(result)
    summary, process, _ = extract_sections(result_text)
    return {
        "summary": summary,
        "sources": web_results,
        "process": process
    }

def create_prompt(query: str, context: str) -> str:
    return (
        "You are a helpful research assistant. Using the provided context below, write a detailed, multi-paragraph summary (at least 8 sentences) answering the research question. "
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
        f"Context:\n{context}\n\n"
        f"Research Question:\n{query}\n"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 