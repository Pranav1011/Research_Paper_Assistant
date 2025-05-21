import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";

interface ResearchFormProps {
  onResearchComplete: (results: {
    summary: string;
    sources: any[];
    process: any[];
    search_query?: string;
    knowledge_gaps?: any[];
    followup_queries?: any[];
  }) => void;
}

export function ResearchForm({ onResearchComplete }: ResearchFormProps) {
  const [topic, setTopic] = useState("");
  const [model, setModel] = useState("llama3");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8001/api/research", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: topic, model }),
      });

      if (!response.ok) {
        throw new Error("Research request failed");
      }

      const data = await response.json();
      // Map backend response to frontend structure
      onResearchComplete({
        summary: data.result || data.summary || "",
        sources: data.sources || [],
        process: data.process || [],
        search_query: data.search_query,
        knowledge_gaps: data.knowledge_gaps,
        followup_queries: data.followup_queries,
      });
      toast.success("Research completed successfully!");
    } catch (error) {
      console.error("Research failed:", error);
      toast.error("Failed to complete research. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Start Research</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="topic" className="text-sm font-medium">
              Research Topic
            </label>
            <Input
              id="topic"
              placeholder="Enter your research topic..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="model" className="text-sm font-medium">
              Select Model
            </label>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="llama2:13b">Llama 2 13B</SelectItem>
                <SelectItem value="mistral">Mistral</SelectItem>
                <SelectItem value="llama3">Llama 3</SelectItem>
                <SelectItem value="deepseek-coder">DeepSeek Coder</SelectItem>
                <SelectItem value="Qwen2">Qwen2</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? "Researching..." : "Start Research"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}