import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface Source {
  title: string;
  href: string;
  body: string;
}

interface ResearchResultsProps {
  summary: string;
  sources: Source[];
  process?: string;
}

function renderSummary(summary: any): React.ReactNode {
  if (summary == null) return <p>No summary available.</p>;

  if (typeof summary === "string" || typeof summary === "number") {
    return <p>{summary}</p>;
  }

  if (Array.isArray(summary)) {
    return (
      <ul className="list-disc pl-5">
        {summary.map((item, idx) => (
          <li key={idx}>{renderSummary(item)}</li>
        ))}
      </ul>
    );
  }

  if (typeof summary === "object") {
    if (
      Object.keys(summary).length === 1 &&
      (summary.description || summary.explanation)
    ) {
      return <p>{summary.description || summary.explanation}</p>;
    }
    return (
      <div className="space-y-4">
        {Object.entries(summary).map(([key, value]) => (
          <div key={key}>
            <h4 className="font-semibold capitalize mb-1">{key.replace(/_/g, " ")}</h4>
            {renderSummary(value)}
          </div>
        ))}
      </div>
    );
  }

  return <p>{String(summary)}</p>;
}

function renderGapsOrQueries(items: any[]) {
  if (!items || items.length === 0) return null;
    return (
      <ul className="list-disc pl-5">
        {items.map((item, idx) => {
          if (typeof item === "string" || typeof item === "number") {
            return <li key={idx}>{item}</li>;
          }
          if (typeof item === "object" && item !== null) {
            return (
              <li key={idx}>
                {item.description || item.gap || item.query || JSON.stringify(item)}
                {item.explanation && (
                  <div className="text-sm text-muted-foreground pl-2">{item.explanation}</div>
                )}
                {item.followup_query && (
                  <div className="text-sm text-muted-foreground pl-2">
                    <strong>Follow-up:</strong> {item.followup_query}
                  </div>
                )}
              </li>
            );
          }
          return <li key={idx}>{String(item)}</li>;
        })}
      </ul>
    );
  }

  export function ResearchResults({
    summary,
    sources = [],
    process = "",
  }: ResearchResultsProps) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Research Results</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="summary" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="summary">Summary</TabsTrigger>
              <TabsTrigger value="sources">Sources</TabsTrigger>
              <TabsTrigger value="process">Process</TabsTrigger>
            </TabsList>
  
            <TabsContent value="summary" className="mt-4">
              <h3 className="font-semibold mb-1">Summary</h3>
              <div className="whitespace-pre-line">{summary || "No summary available."}</div>
            </TabsContent>

            {/* Sources Tab */}
            <TabsContent value="sources" className="mt-4">
              {sources.length > 0 ? (
                <ul className="space-y-4">
                  {sources.map((source, idx) => (
                    <li key={idx} className="border rounded-lg p-4 shadow-sm">
                      <div className="font-semibold mb-2">{source.title || "Untitled"}</div>
                      <div className="text-sm mb-2">{source.body || "No description available."}</div>
                      {source.href && (
                        <a
                          href={source.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline text-sm inline-flex items-center gap-1"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                            <polyline points="15 3 21 3 21 9"></polyline>
                            <line x1="10" y1="14" x2="21" y2="3"></line>
                          </svg>
                          Visit Source
                        </a>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No sources found.</p>
              )}
            </TabsContent>

            {/* Process Tab */}
            <TabsContent value="process" className="mt-4">
              {process ? (
                <div className="prose">
                  {process}
                </div>
              ) : (
                <p>No process steps available.</p>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    );
  } 