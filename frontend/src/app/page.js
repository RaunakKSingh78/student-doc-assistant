'use client';

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const renderFormattedAnswer = (text) => {
  if (!text) return null;

  const lines = text.split("\n");
  const elements = [];
  let currentList = [];

  const parseInlineMarkdown = (line) => {
    // Basic bold parsing: **text**
    const parts = line.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <strong key={index} className="font-semibold text-sky-200">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  lines.forEach((line, index) => {
    const trimmedLine = line.trim();

    // Check if line is a bullet point
    if (trimmedLine.startsWith("* ") || trimmedLine.startsWith("- ")) {
      const content = trimmedLine.substring(2);
      currentList.push(
        <li key={`li-${index}`} className="ml-5 list-disc pl-1 mb-1 text-slate-200">
          {parseInlineMarkdown(content)}
        </li>
      );
    } else {
      // If we were building a list, push it first
      if (currentList.length > 0) {
        elements.push(
          <ul key={`ul-${index}`} className="mb-4 space-y-1">
            {currentList}
          </ul>
        );
        currentList = [];
      }

      if (trimmedLine === "") {
        elements.push(<div key={`br-${index}`} className="h-2" />);
      } else {
        elements.push(
          <p key={`p-${index}`} className="mb-4 leading-relaxed text-slate-200">
            {parseInlineMarkdown(line)}
          </p>
        );
      }
    }
  });

  // Push any remaining list
  if (currentList.length > 0) {
    elements.push(
      <ul key={`ul-end`} className="mb-4 space-y-1">
        {currentList}
      </ul>
    );
  }

  return <div>{elements}</div>;
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [status, setStatus] = useState("Checking backend...");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        setStatus(data.ready ? "Backend ready" : "Backend starting");
      } catch (err) {
        setStatus("Backend unreachable");
      }
    };
    checkHealth();
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || "Query failed");
      }

      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message || "Unable to fetch answer");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="futuristic-bg min-h-screen overflow-hidden text-white">
      <div className="glow-ring"></div>
      <div className="relative z-10 flex min-h-screen flex-col">
        <section className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6 lg:px-10">
          <div className="w-full max-w-5xl rounded-[40px] border border-white/10 bg-[rgba(255,255,255,0.06)] p-10 shadow-[0_60px_140px_rgba(2,10,30,0.45)] backdrop-blur-2xl">
            <div className="space-y-8 text-center">
              <div className="inline-flex rounded-full border border-sky-300/30 bg-sky-300/10 px-5 py-2 text-xs uppercase tracking-[0.35em] text-sky-200 shadow-[0_0_40px_rgba(56,189,248,0.18)]">
                  Docs AI
              </div>
              <div>
                <h1 className="text-5xl font-semibold tracking-tight text-white sm:text-6xl">
                  Hey Armando!
                  <span className="block text-transparent bg-clip-text bg-gradient-to-r from-sky-300 via-fuchsia-300 to-emerald-300">
                    Can I help you with anything?
                  </span>
                </h1>
                <p className="mx-auto mt-6 max-w-3xl text-base leading-8 text-slate-300 sm:text-lg">
                  Please type your query below.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="mx-auto flex w-full max-w-3xl flex-col gap-4 sm:flex-row">
                <input
                  type="text"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Ask anything you need"
                  className="w-full rounded-full border border-white/10 bg-slate-950/70 px-6 py-4 text-base text-white placeholder:text-slate-500 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] outline-none transition duration-300 focus:border-sky-400 focus:ring-2 focus:ring-sky-400/20"
                  required
                />
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="gradiant-button inline-flex h-14 items-center justify-center rounded-full px-8 text-sm font-semibold text-white shadow-[0_20px_50px_rgba(56,189,248,0.24)] transition duration-300 hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Sending..." : "Send"}
                </button>
              </form>

              <div className="mt-4 flex flex-wrap justify-center gap-3">
                {["Accommodation Rules", "Hostel Rules", "Penalty Points", "PhD and PG Rules", "UG Rules"].map((label) => (
                  <span
                    key={label}
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 shadow-[0_10px_30px_rgba(15,23,42,0.25)]"
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto w-full max-w-6xl px-4 pb-16 sm:px-6 lg:px-10">
          <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <div className="panel-glow rounded-[32px] border border-white/10 bg-[rgba(255,255,255,0.04)] p-8 shadow-[0_30px_90px_rgba(2,10,30,0.2)]">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-sky-300">Answer</p>
                  <h2 className="mt-2 text-3xl font-semibold text-white">Response</h2>
                </div>
                <span className="rounded-full bg-slate-900/80 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-slate-300">
                  {loading ? "Loading" : "Ready"}
                </span>
              </div>
              <div className="mt-8 min-h-[240px] rounded-[28px] border border-slate-700/80 bg-slate-950/75 p-6 text-sm leading-7 text-slate-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]">
                {loading ? (
                  <p>Loading answer…</p>
                ) : answer ? (
                  renderFormattedAnswer(answer)
                ) : (
                  <p className="text-slate-500">Scroll down to view the answer. Ask a question to populate this section.</p>
                )}
              </div>
            </div>

            <div className="panel-glow rounded-[32px] border border-white/10 bg-[rgba(255,255,255,0.04)] p-8 shadow-[0_30px_90px_rgba(2,10,30,0.2)]">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-fuchsia-300">Source</p>
                  <h2 className="mt-2 text-3xl font-semibold text-white">Retrieved passages</h2>
                </div>
                <span className="rounded-full bg-slate-900/80 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-slate-300">
                  {sources.length} items
                </span>
              </div>

              <div className="mt-8 space-y-4">
                {sources.length === 0 ? (
                  <div className="rounded-[28px] border border-dashed border-slate-700/80 bg-slate-950/75 p-6 text-sm text-slate-500">
                    No passages available yet. After submitting your query, the retrieved sources appear here.
                  </div>
                ) : (
                  sources.map((source, index) => (
                    <article
                      key={`${source.source}-${index}`}
                      className="rounded-[28px] border border-slate-700/80 bg-slate-950/75 p-5"
                    >
                      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <p className="text-sm font-semibold text-white">{source.source || "Unknown source"}</p>
                        <span className="rounded-full bg-sky-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-sky-200">
                          Passage {index + 1}
                        </span>
                      </div>
                      <p className="whitespace-pre-wrap text-sm leading-7 text-slate-300">{source.text}</p>
                    </article>
                  ))
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
