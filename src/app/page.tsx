"use client";

import { useState, useRef } from "react";
import { ResearchForm } from "@/components/research-form";
import { ResearchResults } from "@/components/research-results";

export default function Home() {
  const [result, setResult] = useState("");
  const [webSummary, setWebSummary] = useState("");
  const [webSources, setWebSources] = useState<any[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedResult, setStreamedResult] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Handler for research completion (non-streaming fallback)
  const handleResearchComplete = (results: { result: string; webSummary?: string; webSources?: any[] }) => {
    setResult(results.result);
    setWebSummary(results.webSummary || "");
    setWebSources(results.webSources || []);
    setStreamedResult("");
    setIsStreaming(false);
    setHasSearched(true);
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  // Handler for streaming output (if backend supports it)
  const handleStream = async (formData: FormData) => {
    setIsStreaming(true);
    setStreamedResult("");
    setResult("");
    setWebSummary("");
    setWebSources([]);
    setHasSearched(true);
    try {
      const response = await fetch("http://localhost:8000/api/research/stream", {
        method: "POST",
        body: formData,
      });
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      let text = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = new TextDecoder().decode(value);
        text += chunk;
        setStreamedResult(text);
      }
      setIsStreaming(false);
      setResult(text);
    } catch (err) {
      setIsStreaming(false);
      setStreamedResult("");
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#18181b] text-white dark relative overflow-x-hidden">
      {/* Minimalist background pattern/gradient */}
      <div className="pointer-events-none fixed inset-0 z-0 bg-gradient-to-br from-[#23243a] via-[#18181b] to-[#23243a] opacity-80" style={{backgroundImage: 'radial-gradient(circle at 60% 40%, rgba(80,80,160,0.08) 0, transparent 60%), radial-gradient(circle at 20% 80%, rgba(80,160,160,0.07) 0, transparent 70%)'}} />
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Title and caption (centered, only if not searched) */}
        {!hasSearched && (
          <div className="flex flex-col items-center justify-center py-12">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">Re-Searcher</h1>
            <p className="text-lg md:text-xl text-zinc-400 text-center max-w-2xl">
              Your aid to deep dive into your research paper and its topics
            </p>
          </div>
        )}
        {/* Title above results when searched */}
        {hasSearched && (
          <div className="w-full flex justify-center pt-8 pb-2">
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Re-Searcher</h1>
          </div>
        )}
        <div className="flex-1 flex flex-col-reverse md:flex-col-reverse lg:flex-col-reverse max-w-5xl mx-auto w-full px-2 md:px-6">
          {/* Research Form at the bottom */}
          <div className="sticky bottom-0 z-10 bg-transparent pt-4 pb-2 w-full">
            <div className="w-full">
              <ResearchForm
                onResearchComplete={handleResearchComplete}
                // Optionally pass handleStream for streaming support
              />
            </div>
          </div>
          {/* Results above the form */}
          <div ref={resultsRef} className="flex-1 mb-4 w-full">
            {(isStreaming || streamedResult) ? (
              <div className="mb-6 w-full">
                <div className="text-lg font-semibold mb-2">Research Results (Streaming)</div>
                <div className="prose prose-sm max-w-none bg-zinc-900 text-zinc-200 p-4 rounded-lg shadow-sm whitespace-pre-line">
                  {streamedResult || <span className="text-zinc-400">Waiting for response...</span>}
                </div>
              </div>
            ) : (
              <div className="w-full">
                <ResearchResults result={result} webSummary={webSummary} webSources={webSources} />
              </div>
            )}
          </div>
        </div>
        {/* Footer */}
        <footer className="w-full text-center py-4 text-zinc-500 text-xs opacity-80 mt-8">
          Developed by Sai Pranav
        </footer>
      </div>
    </div>
  );
}