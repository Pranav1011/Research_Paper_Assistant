"use client";

import { useState } from "react";
import { ResearchForm } from "@/components/research-form";
import { ResearchResults } from "@/components/research-results";

export default function Home() {
  const [result, setResult] = useState("");
  const [webSummary, setWebSummary] = useState("");
  const [webSources, setWebSources] = useState<any[]>([]);

  const handleResearchComplete = (results: { result: string; webSummary?: string; webSources?: any[] }) => {
    setResult(results.result);
    setWebSummary(results.webSummary || "");
    setWebSources(results.webSources || []);
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">Deep Researcher TS</h1>
          <p className="text-muted-foreground">
            A modern web research assistant powered by local LLMs
          </p>
        </div>
        <ResearchForm onResearchComplete={handleResearchComplete} />
        <ResearchResults result={result} webSummary={webSummary} webSources={webSources} />
      </div>
    </div>
  );
}