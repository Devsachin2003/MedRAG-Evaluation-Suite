import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl } from "../config/api";

export type SourceChunk = {
  text: string;
  source: string;
  chunk_index?: number | null;
  distance?: number | null;
};

type Role = "user" | "assistant";

type ChatTurn = {
  id: string;
  role: Role;
  content: string;
  sources?: SourceChunk[];
};

function parseSseData(line: string): unknown | null {
  const prefix = "data: ";
  if (!line.startsWith(prefix)) return null;
  try {
    return JSON.parse(line.slice(prefix.length));
  } catch {
    return null;
  }
}

export function Chat() {
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  useEffect(() => () => requestRef.current?.abort(), []);

  const scrollToBottom = () => {
    requestAnimationFrame(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }));
  };

  const send = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || streaming) return;

    const userTurn: ChatTurn = { id: crypto.randomUUID(), role: "user", content: trimmed };
    const assistantId = crypto.randomUUID();
    const nextHistory = [...messages, userTurn];

    setMessages([...nextHistory, { id: assistantId, role: "assistant", content: "", sources: [] }]);
    setInput("");
    setStreaming(true);
    scrollToBottom();

    const payload = {
      messages: [{ role: "user", content: trimmed }],
    };

    let assistantContent = "";
    let sources: SourceChunk[] = [];
    const controller = new AbortController();
    requestRef.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 180000);
    let completed = false;

    try {
      const res = await fetch(apiUrl("/api/chat/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const parsed = parseSseData(line) as { type?: string; token?: string; sources?: SourceChunk[] } | null;
          if (!parsed || typeof parsed !== "object") continue;
          if (parsed.type === "error") throw new Error("Generation failed");
          if (parsed.type === "done") completed = true;

          if (parsed.type === "sources" && Array.isArray(parsed.sources)) {
            sources = parsed.sources;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, sources: [...parsed.sources!] } : m))
            );
          }
          if (parsed.type === "token" && typeof parsed.token === "string") {
            assistantContent += parsed.token;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: assistantContent } : m))
            );
            scrollToBottom();
          }
        }
      }
      if (!completed) throw new Error("Incomplete stream");
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: "Could not reach the chat endpoint. Check that the backend API is running and reachable.",
                sources,
              }
            : m
        )
      );
    } finally {
      window.clearTimeout(timeout);
      controller.abort();
      requestRef.current = null;
      setStreaming(false);
      scrollToBottom();
    }
  }, [input, messages, streaming]);

  return (
    <section className="flex h-[calc(100vh-12rem)] min-h-[420px] flex-col rounded-2xl border border-slate-200/80 bg-white shadow-soft">
      <div className="border-b border-slate-100 px-5 py-4">
        <h2 className="font-display text-lg font-semibold text-ink-900">RAG chat</h2>
        <p className="text-sm text-slate-600">Answers use retrieved chunks from your indexed documents.</p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4 scrollbar-thin sm:px-6">
        {messages.length === 0 && (
          <div className="rounded-xl bg-slate-50 px-4 py-8 text-center text-sm text-slate-600">
            Ask a clinical or policy question grounded in your uploads — for example:{" "}
            <span className="italic text-slate-800">
              “Summarize contraindications mentioned in the uploaded guideline.”
            </span>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[min(100%,42rem)] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                m.role === "user"
                  ? "bg-clinical-600 text-white"
                  : "bg-slate-50 text-ink-900 ring-1 ring-slate-100"
              }`}
            >
              <p className="whitespace-pre-wrap">{m.content || (streaming ? "…" : "")}</p>

              {m.role === "assistant" && (m.sources?.length ?? 0) > 0 && (
                <details className="mt-3 border-t border-slate-200/80 pt-3">
                  <summary className="cursor-pointer select-none text-xs font-semibold uppercase tracking-wide text-clinical-700 hover:text-clinical-800">
                    View sources ({m.sources!.length} chunks)
                  </summary>
                  <ul className="mt-2 space-y-2">
                    {m.sources!.map((s, i) => (
                      <li
                        key={`${m.id}-src-${i}`}
                        className="rounded-lg bg-white p-3 text-xs text-slate-700 ring-1 ring-slate-100"
                      >
                        <p className="font-medium text-slate-500">
                          {s.source}
                          {s.chunk_index != null ? ` · chunk ${s.chunk_index}` : ""}
                          {s.distance != null ? ` · distance ${s.distance.toFixed(4)}` : ""}
                        </p>
                        <p className="mt-1 whitespace-pre-wrap font-mono text-[11px] leading-snug text-slate-800">
                          {s.text}
                        </p>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-100 p-4 sm:p-5">
        <div className="flex gap-2">
          <textarea
            maxLength={8000}
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Ask a question…"
            className="min-h-[52px] flex-1 resize-none rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none ring-clinical-500/30 placeholder:text-slate-400 focus:border-clinical-400 focus:ring-2"
            disabled={streaming}
          />
          <button
            type="button"
            onClick={() => void send()}
            disabled={streaming || !input.trim()}
            className="self-end rounded-xl bg-clinical-600 px-5 py-3 text-sm font-semibold text-white shadow-md shadow-clinical-600/20 transition hover:bg-clinical-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {streaming ? "…" : "Send"}
          </button>
        </div>
      </div>
    </section>
  );
}
