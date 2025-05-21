"use client";

import { useState } from "react";
import { ResearchForm } from "@/components/research-form";
import { ResearchResults } from "@/components/research-results";

interface ResearchData {
  summary: string;
  sources: string[];
  process: string[];
}

export default function Home() {
  const [researchData, setResearchData] = useState<ResearchData>({
    summary: "",
    sources: [],
    process: [],
  });

  const handleResearchComplete = (results: ResearchData) => {
    setResearchData(results);
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
        <ResearchResults {...researchData} />
      </div>
    </div>
  );
}