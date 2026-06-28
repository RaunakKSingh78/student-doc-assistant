'use client';

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
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
        body: JSON.stringify({ query, top_k: topK }),
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
    <div className="min-h-screen bg-slate-50 text-slate-900 px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl space-y-8">
        <section className="rounded-3xl bg-white p-8 shadow-2xl shadow-slate-200">
          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium uppercase tracking-[0.24em] text-sky-600">
                Student Document Assistant
              </p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
                Ask questions and inspect the retrieved passages.
              </h1>
              <p className="mt-4 max-w-3xl text-base text-slate-600 sm:text-lg">
                Enter a query, then review the AI answer with the documents that were used to generate it.
              </p>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-900">Backend status</p>
                  <p className="text-sm text-slate-600">{status}</p>
                </div>
                <div className="flex items-center gap-3">
                  <label className="text-sm font-medium text-slate-700" htmlFor="topK">
                    Top K
                  </label>
                  <select
                    id="topK"
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                    className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-500"
                  >
                    {[3, 5, 7, 10].map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="mt-6 space-y-4">
                <label className="block text-sm font-medium text-slate-700" htmlFor="query">
                  Ask a question about student documents
                </label>
                <textarea
                  id="query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  rows={5}
                  placeholder="Example: What are the hostel penalty rules for late checkout?"
                  className="w-full rounded-3xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-sky-500"
                  required
                />

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="inline-flex items-center justify-center rounded-full bg-sky-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-400"
                  >
                    {loading ? "Generating answer…" : "Get answer"}
                  </button>
                  <p className="text-sm text-slate-500">
                    {error ? `Error: ${error}` : "Responses are based on document chunks and retrieved passages."}
                  </p>
                </div>
              </form>
            </div>
          </div>
        </section>

        <section className="grid gap-8 lg:grid-cols-[1.4fr_0.9fr]">
          <div className="rounded-3xl bg-white p-8 shadow-2xl shadow-slate-200">
            <h2 className="text-xl font-semibold text-slate-900">Answer</h2>
            <div className="mt-6 min-h-[220px] rounded-3xl border border-slate-200 bg-slate-50 p-6 text-sm leading-7 text-slate-700">
              {loading ? (
                <p>Loading answer…</p>
              ) : answer ? (
                <p>{answer}</p>
              ) : (
                <p className="text-slate-500">Submit a question to see the generated answer here.</p>
              )}
            </div>
          </div>

          <div className="rounded-3xl bg-white p-8 shadow-2xl shadow-slate-200">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-slate-900">Retrieved passages</h2>
                <p className="text-sm text-slate-500">Displays sources and excerpts used to answer your question.</p>
              </div>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                {sources.length} items
              </span>
            </div>

            <div className="mt-6 space-y-4">
              {sources.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
                  No passages available yet. Do a query to show retrieved context.
                </div>
              ) : (
                sources.map((source, index) => (
                  <article
                    key={`${source.source}-${index}`}
                    className="rounded-3xl border border-slate-200 bg-slate-50 p-5"
                  >
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <p className="text-sm font-semibold text-slate-900">{source.source || "Unknown source"}</p>
                      <span className="rounded-full bg-slate-200 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-600">
                        Passage {index + 1}
                      </span>
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{source.text}</p>
                  </article>
                ))
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
