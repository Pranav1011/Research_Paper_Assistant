import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { Paperclip, Globe2 } from "lucide-react";

interface ResearchFormProps {
  onResearchComplete: (results: {
    result: string;
    webSummary?: string;
    webSources?: any[];
  }) => void;
}

export function ResearchForm({ onResearchComplete }: ResearchFormProps) {
  const [topic, setTopic] = useState("");
  const [model, setModel] = useState("llama3");
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [coliResult, setColiResult] = useState<string>("");
  const [webSummary, setWebSummary] = useState<string>("");
  const [webSources, setWebSources] = useState<any[]>([]);
  const [includeWeb, setIncludeWeb] = useState(false);

  const handleColiVaraSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Please upload a PDF file for ColiVara research.");
      return;
    }
    setIsLoading(true);
    setWebSummary("");
    setWebSources([]);
    try {
      const formData = new FormData();
      formData.append('query', topic);
      if (file) {
        formData.append('file', file);
      }
      const response = await fetch("http://localhost:8000/api/research", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Research request failed");
      }
      const data = await response.json();
      setColiResult(data.summary || data.result || "");
      onResearchComplete({ result: data.summary || data.result || "" });
      toast.success("ColiVara research completed!");
      if (includeWeb) {
        await handleWebSearch(true, data.summary || data.result || "");
      }
    } catch (error) {
      console.error("Research failed:", error);
      toast.error("Failed to complete research. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleWebSearch = async (auto = false, baseResult = "") => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/websearch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: topic, model }),
      });
      if (!response.ok) {
        throw new Error("Web search request failed");
      }
      const data = await response.json();
      setWebSummary(data.summary || "");
      setWebSources(data.sources || []);
      onResearchComplete({ result: auto ? baseResult : coliResult, webSummary: data.summary || "", webSources: data.sources || [] });
      toast.success("Web search completed!");
    } catch (error) {
      console.error("Web search failed:", error);
      toast.error("Failed to complete web search. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full bg-zinc-900 text-zinc-200 border-zinc-800">
      <CardContent>
        <form onSubmit={handleColiVaraSubmit} className="space-y-3">
          <div className="space-y-1">
            <label htmlFor="topic" className="text-sm font-medium">
              Research Topic
            </label>
            <Input
              id="topic"
              placeholder="Enter your research topic..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              required
              className="bg-zinc-800 border-zinc-700 text-zinc-100"
            />
          </div>

          {/* PDF and Web Search side by side, justified between */}
          <div className="flex items-center gap-4">
            {/* PDF Attach */}
            <div className="flex items-center gap-2">
              <label htmlFor="file" className="text-sm font-medium flex items-center gap-1">
                <span className="relative">
                  <Paperclip className="w-5 h-5 text-blue-400" />
                  <span className="absolute -top-1 -right-2 text-xs text-red-500 font-bold">*</span>
                </span>
                <span className="sr-only">Upload PDF (Required)</span>
              </label>
              <Input
                id="file"
                type="file"
                accept=".pdf"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="w-28 h-8 p-0 border-none bg-zinc-800 text-zinc-100 file:bg-zinc-700 file:text-zinc-100 file:rounded file:px-2 file:py-1 file:border-none"
                required
              />
              <span className="text-xs text-zinc-400 ml-2">PDF required</span>
            </div>
            {/* Web Search */}
            <div className="flex items-center space-x-2 ml-2 flex-shrink-0">
              <input
                id="includeWeb"
                type="checkbox"
                checked={includeWeb}
                onChange={(e) => setIncludeWeb(e.target.checked)}
                className="form-checkbox h-4 w-4 text-blue-600"
              />
              <label htmlFor="includeWeb" className="text-sm font-medium flex items-center gap-1">
                <Globe2 className="w-4 h-4 text-blue-400" />
                Web Search
              </label>
            </div>
          </div>

          {includeWeb && (
            <div className="space-y-1">
              <label htmlFor="model" className="text-sm font-medium">
                Select Model (Web Search Only)
              </label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-zinc-100">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-700 text-zinc-100">
                  <SelectItem value="llama3">Llama 3</SelectItem>
                  <SelectItem value="Qwen2">Qwen2</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-400">
                Model selection only applies when web search is enabled
              </p>
            </div>
          )}

          <Button type="submit" className="w-full mt-2" disabled={isLoading || !file}>
            {isLoading ? "Researching..." : "Start Research"}
          </Button>
        </form>
        {coliResult && !includeWeb && (
          <Button onClick={() => handleWebSearch()} className="w-full mt-2" disabled={isLoading} variant="secondary">
            {isLoading ? "Searching Web..." : "Run Web Search"}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}