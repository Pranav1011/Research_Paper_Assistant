import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ReactMarkdown from "react-markdown";

interface Source {
  title: string;
  href: string;
  body: string;
  page_image?: string;
  page_number?: string;
}

interface ResearchResultsProps {
  result: any; // Accept any type for raw debug
  webSummary?: string; // Web search summary
  webSources?: Source[]; // Web search sources
}

// Helper to extract sections from answer text
function parseSections(answer: string) {
  if (!answer) return [];
  // Remove extra quotes and normalize
  let clean = answer.replace(/^"|"$/g, "").replace(/\\n/g, "\n");
  // Split by section headers
  const sectionRegex = /^(Summary|Key Findings|Trends in Industry|Future Trends|Process|Sources|Results):/gim;
  let parts = clean.split(sectionRegex).filter(Boolean);
  let sections = [];
  if (parts.length === 1) {
    // Only one section, just return as results
    let content = parts[0].trim();
    // If it starts with 'Summary:' or 'Results:', strip it
    if (content.toLowerCase().startsWith("summary:")) {
      content = content.slice(8).trim();
    } else if (content.toLowerCase().startsWith("results:")) {
      content = content.slice(8).trim();
    }
    sections.push({ title: "Results", content });
  } else {
    for (let i = 0; i < parts.length - 1; i += 2) {
      let title = parts[i].trim();
      if (title.toLowerCase() === "summary") title = "Results";
      sections.push({ title, content: parts[i + 1].trim() });
    }
  }
  return sections;
}

// Improved bullet formatting for specific sections
function preprocessBullets(text: string, sectionTitle: string) {
  // Only apply to these sections
  const bulletSections = ["key findings", "trends in industry", "future trends"];
  if (bulletSections.includes(sectionTitle.toLowerCase())) {
    // If already markdown bullets, return as is
    if (/^\s*[-*] /m.test(text)) return text;
    // Try splitting on semicolons first
    if (text.includes(";")) {
      return text
        .split(";")
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => (line.startsWith("- ") || line.startsWith("* ")) ? line : `- ${line}`)
        .join("\n");
    }
    // Try splitting on newlines
    if (text.includes("\n")) {
      return text
        .split("\n")
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => (line.startsWith("- ") || line.startsWith("* ")) ? line : `- ${line}`)
        .join("\n");
    }
    // Try splitting on periods (for sentences)
    if (text.includes(". ")) {
      return text
        .split(/\.\s+/)
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => (line.endsWith(".")) ? line : line + ".")
        .map(line => (line.startsWith("- ") || line.startsWith("* ")) ? line : `- ${line}`)
        .join("\n");
    }
    // Try splitting on commas (last resort)
    if (text.includes(",")) {
      return text
        .split(",")
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => (line.startsWith("- ") || line.startsWith("* ")) ? line : `- ${line}`)
        .join("\n");
    }
  }
  // For other sections, return as is
  return text;
}

export function ResearchResults({ result, webSummary = "", webSources = [] }: ResearchResultsProps) {
  // Try to extract and format the answer
  let answer = result?.summary || result?.result || result || "";
  // If answer is an object, try to get a string
  if (typeof answer === "object") {
    answer = JSON.stringify(answer, null, 2);
  }
  let sections = parseSections(answer);
  // Remove the 'Sources' section from the results tab
  sections = sections.filter(section => section.title.toLowerCase() !== "sources");

  return (
    <Card className="w-full bg-zinc-900 text-zinc-200">
      <CardHeader>
        <CardTitle className="text-zinc-100">Research Results</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="result" className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-zinc-800">
            <TabsTrigger value="result">Results</TabsTrigger>
            <TabsTrigger value="web">Web Search</TabsTrigger>
            <TabsTrigger value="sources">Web Search Sources</TabsTrigger>
          </TabsList>

          {/* Beautifully formatted answer */}
          <TabsContent value="result" className="mt-4">
            {sections.length > 0 && sections[0].content ? (
              <div className="space-y-6">
                {sections.map((section, idx) => (
                  <div key={idx}>
                    {sections.length > 1 && (
                      <h3 className="text-lg font-semibold mb-2 text-zinc-100 border-b border-zinc-700 pb-1 dark:text-zinc-100">{section.title}</h3>
                    )}
                    <div className="prose prose-sm max-w-none bg-zinc-800 text-zinc-200 p-4 rounded-lg shadow-sm dark:text-zinc-200">
                      <ReactMarkdown>{preprocessBullets(section.content, section.title)}</ReactMarkdown>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-zinc-400 italic">No results available. Try a different PDF or research topic.</div>
            )}
            {/* Debug: Raw JSON */}
            <details className="mt-6">
              <summary className="cursor-pointer text-xs text-zinc-400">Show Raw JSON</summary>
              <pre className="text-xs bg-zinc-900 p-2 rounded overflow-x-auto mt-2 text-zinc-300">{JSON.stringify(result, null, 2)}</pre>
            </details>
          </TabsContent>

          {/* Web Search Summary */}
          <TabsContent value="web" className="mt-4">
            <div className="whitespace-pre-line break-words prose max-w-none bg-zinc-800 text-zinc-200 p-4 rounded-lg shadow-sm dark:text-zinc-200">
              <ReactMarkdown>{webSummary || "No web search performed yet."}</ReactMarkdown>
            </div>
          </TabsContent>

          {/* Web Search Sources */}
          <TabsContent value="sources" className="mt-4">
            {webSources.length > 0 ? (
              <ul className="space-y-4">
                {webSources.map((source, idx) => (
                  <li key={idx} className="border rounded-lg p-4 shadow-sm bg-zinc-800 text-zinc-200">
                    <div className="font-semibold mb-2 text-zinc-100">{source.title || "Untitled"}</div>
                    <div className="text-sm mb-2 text-zinc-200">{source.body || "No description available."}</div>
                    {source.page_image && (
                      <div className="mt-2 mb-2">
                        <img
                          src={source.page_image}
                          alt={`Page ${source.page_number || idx + 1}`}
                          className="max-w-full h-auto rounded-lg shadow-sm"
                        />
                        {source.page_number && (
                          <p className="text-sm text-zinc-400 mt-1">
                            Page {source.page_number}
                          </p>
                        )}
                      </div>
                    )}
                    {source.href && (
                      <a
                        href={source.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:underline text-sm inline-flex items-center gap-1"
                      >
                        Visit Source
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-zinc-400 italic">No sources found.</p>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 