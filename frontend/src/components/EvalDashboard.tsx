import { useMemo, useState } from "react";
import { apiUrl } from "../config/api";

type EvalSummary = {
  context_precision: number | null;
  faithfulness: number | null;
  answer_relevancy: number | null;
  output_csv?: string;
};

type TestCaseRow = {
  case_id: string;
  question: string;
  ground_truth: string;
  generated_answer: string;
  context_precision?: number;
  faithfulness?: number;
  answer_relevancy?: number;
  retrieved_contexts?: string[];
};

const EMPTY_SUMMARY: EvalSummary = {
  context_precision: null,
  faithfulness: null,
  answer_relevancy: null,
};

function MetricCard({
  title,
  subtitle,
  value,
  suffix,
  barPct,
  tone,
}: {
  title: string;
  subtitle: string;
  value: number | null;
  suffix: string;
  barPct: number | null;
  tone: "sky" | "emerald" | "amber";
}) {
  const tones = {
    sky: "from-sky-500 to-clinical-600",
    emerald: "from-emerald-500 to-teal-600",
    amber: "from-amber-400 to-orange-500",
  } as const;

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-soft">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</p>
      <p className="mt-1 text-sm text-slate-600">{subtitle}</p>
      <div className="mt-4 flex items-baseline gap-1">
        <span className="font-display text-4xl font-bold tabular-nums text-ink-900">{value ?? "N/A"}</span>
        <span className="text-lg font-medium text-slate-500">{value == null ? "" : suffix}</span>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${tones[tone]} transition-[width] duration-700`}
          style={{ width: `${Math.min(100, Math.max(0, barPct ?? 0))}%` }}
        />
      </div>
    </div>
  );
}

function SourceList({ sources }: { sources?: string[] }) {
  if (!sources || sources.length === 0) {
    return <p className="text-xs text-slate-500">No chunks returned.</p>;
  }

  return (
    <details>
      <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-clinical-700 hover:text-clinical-800">
        View chunks ({sources.length})
      </summary>
      <ul className="mt-2 space-y-2">
        {sources.map((chunk, index) => (
          <li key={`${index}-${chunk.slice(0, 24)}`} className="rounded-lg bg-slate-50 p-3 text-xs text-slate-700">
            {chunk}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function EvalDashboard() {
  const [summary, setSummary] = useState<EvalSummary>(EMPTY_SUMMARY);
  const [cases, setCases] = useState<TestCaseRow[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const percent = useMemo(() => {
    const toPct = (value: number | null) => value == null || !Number.isFinite(value)
      ? null : Math.round(Math.max(0, Math.min(1, value)) * 100);
    return {
      contextPrecision: toPct(summary.context_precision),
      faithfulness: toPct(summary.faithfulness),
      answerRelevancy: toPct(summary.answer_relevancy),
    };
  }, [summary]);

  const runEvaluation = async () => {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch(apiUrl("/api/evaluation/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = (await response.json()) as { summary?: EvalSummary; cases?: TestCaseRow[]; output_csv?: string };
      if (payload.summary) setSummary({ ...payload.summary, output_csv: payload.output_csv });
      if (Array.isArray(payload.cases)) setCases(payload.cases);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="space-y-8">
      <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-soft">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="font-display text-xl font-bold text-ink-900">Evaluation dashboard</h2>
            <p className="mt-1 text-sm text-slate-600">
              Run the ragas harness against the live MedRAG pipeline and review the resulting scores and chunks.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void runEvaluation()}
            disabled={running}
            className="rounded-xl bg-clinical-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-clinical-600/20 transition hover:bg-clinical-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {running ? "Running..." : "Run evaluation"}
          </button>
        </div>
        {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
        {summary.output_csv && <p className="mt-3 text-xs text-slate-500">CSV saved to: {summary.output_csv}</p>}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <MetricCard
          title="Context precision"
          subtitle="Signal vs. noise in retrieved chunks"
          value={percent.contextPrecision}
          suffix="%"
          barPct={percent.contextPrecision}
          tone="sky"
        />
        <MetricCard
          title="Answer faithfulness"
          subtitle="How grounded the response is"
          value={percent.faithfulness}
          suffix="%"
          barPct={percent.faithfulness}
          tone="emerald"
        />
        <MetricCard
          title="Answer relevancy"
          subtitle="How well the answer addresses the question"
          value={percent.answerRelevancy}
          suffix="%"
          barPct={percent.answerRelevancy}
          tone="amber"
        />
      </div>

      <div className="overflow-hidden rounded-2xl border border-slate-200/80 bg-white shadow-soft">
        <div className="border-b border-slate-100 px-5 py-4">
          <h3 className="font-display text-lg font-semibold text-ink-900">Test cases</h3>
          <p className="text-sm text-slate-600">PPCM-focused regression prompts with retrieved chunks and ragas scores.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1100px] text-left text-sm">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50/80 text-xs font-semibold uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3">Question</th>
                <th className="px-4 py-3">Expected ground truth</th>
                <th className="px-4 py-3">Generated answer</th>
                <th className="px-4 py-3">Context precision</th>
                <th className="px-4 py-3">Faithfulness</th>
                <th className="px-4 py-3">Relevancy</th>
                <th className="px-4 py-3">Chunks</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {cases.length === 0 ? (
                <tr>
                  <td className="px-4 py-6 text-slate-500" colSpan={7}>
                    Click “Run evaluation” to generate scores.
                  </td>
                </tr>
              ) : (
                cases.map((row) => (
                  <tr key={row.case_id} className="hover:bg-slate-50/50">
                    <td className="max-w-xs px-4 py-3 align-top text-slate-800">{row.question}</td>
                    <td className="max-w-xs px-4 py-3 align-top text-slate-600">{row.ground_truth}</td>
                    <td className="max-w-xs px-4 py-3 align-top text-slate-700">{row.generated_answer}</td>
                    <td className="px-4 py-3 align-top text-slate-700">{row.context_precision == null ? "N/A" : `${Math.round(row.context_precision * 100)}%`}</td>
                    <td className="px-4 py-3 align-top text-slate-700">{row.faithfulness == null ? "N/A" : `${Math.round(row.faithfulness * 100)}%`}</td>
                    <td className="px-4 py-3 align-top text-slate-700">{row.answer_relevancy == null ? "N/A" : `${Math.round(row.answer_relevancy * 100)}%`}</td>
                    <td className="px-4 py-3 align-top">
                      <SourceList sources={row.retrieved_contexts} />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
