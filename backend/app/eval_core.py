"""Shared MedRAG evaluation helpers for the CLI and the API."""

from __future__ import annotations

import logging
import os
import math
import tempfile
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from chromadb.utils import embedding_functions
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

from app.rag_core import generate_grounded_response, retrieve_context
from app import settings

try:
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
except ImportError as exc:  # pragma: no cover - import guard for user guidance
    raise RuntimeError(
        "ragas is not installed. Run: pip install ragas datasets pandas requests python-dotenv langchain-groq"
    ) from exc


logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = settings.OUTPUT_DIR / "ragas_evaluation_results.csv"

TEST_CASES: list[dict[str, str]] = [
    {
        "id": "ppcm-contraindications",
        "question": "What are the contraindications or situations where a wearable cardioverter-defibrillator (WCD) is not recommended for PPCM patients?",
        "ground_truth": (
            "The answer should describe contraindications or situations where WCD is not recommended in PPCM, "
            "including any limitations, patient-selection boundaries, or cases where the device should not be used."
        ),
    },
    {
        "id": "ppcm-symptoms",
        "question": "What symptoms are commonly associated with peripartum cardiomyopathy (PPCM)?",
        "ground_truth": (
            "Common PPCM symptoms include shortness of breath, fatigue, orthopnea, paroxysmal nocturnal dyspnea, "
            "edema, palpitations, and signs of heart failure."
        ),
    },
    {
        "id": "ppcm-treatment",
        "question": "What general treatment approach is typically used for PPCM after diagnosis?",
        "ground_truth": (
            "Treatment usually includes standard heart-failure therapy adjusted for pregnancy or postpartum status, "
            "plus close cardiology follow-up and supportive management."
        ),
    },
]


class ChromaDefaultEmbeddings(Embeddings):
    """Adapter that reuses Chroma's default embedding function for ragas evaluation."""

    def __init__(self) -> None:
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_fn(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_fn([text])[0]


def load_env() -> None:
    """Load environment variables from the backend .env file when present."""

    load_dotenv(BACKEND_ROOT / ".env")


def query_chat_api(api_url: str, question: str) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/chat",
        json={"query": question},
        timeout=(10, 180),
        headers={"Authorization": f"Bearer {os.getenv('MEDRAG_API_KEY', '')}"},
    )
    response.raise_for_status()
    payload = response.json()

    sources = payload.get("sources") or []
    retrieved_contexts = [
        source.get("text", "")
        for source in sources
        if isinstance(source, dict) and source.get("text")
    ]

    return {
        "answer": payload.get("response", ""),
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
    }


def query_local_pipeline(question: str) -> dict[str, Any]:
    contexts = retrieve_context(question, k=6)
    answer = generate_grounded_response(question, contexts)
    return {
        "answer": answer,
        "sources": contexts,
        "retrieved_contexts": [
            context.get("text", "") for context in contexts if context.get("text")
        ],
    }


def build_runs(
    question_runner: Callable[[str], dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for case in TEST_CASES:
        logger.info("Evaluating %s", case["id"])
        result = question_runner(case["question"])
        runs.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "ground_truth": case["ground_truth"],
                "generated_answer": result["answer"],
                "retrieved_contexts": result["retrieved_contexts"],
                "retrieved_sources_json": result["sources"],
            }
        )
    return runs


def build_ragas_dataset(runs: list[dict[str, Any]]) -> Dataset:
    records: list[dict[str, Any]] = []
    for run in runs:
        contexts = [
            context
            for context in run["retrieved_contexts"]
            if isinstance(context, str) and context.strip()
        ]
        ground_truth = run["ground_truth"]
        answer = run["generated_answer"]
        question = run["question"]

        records.append(
            {
                "case_id": run["case_id"],
                "question": question,
                "user_input": question,
                "answer": answer,
                "response": answer,
                "contexts": contexts,
                "retrieved_contexts": contexts,
                "ground_truth": ground_truth,
                "reference": ground_truth,
                "ground_truths": [ground_truth],
            }
        )

    return Dataset.from_list(records)


def evaluate_with_ragas(dataset: Dataset) -> pd.DataFrame:
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for evaluation")

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            api_key=groq_api_key,
            model=groq_model,
            temperature=0.0,
            streaming=False,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=1,
            max_tokens=2048,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(ChromaDefaultEmbeddings())

    from ragas.run_config import RunConfig

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=120, max_retries=1, max_workers=2),
    )

    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if isinstance(result, pd.DataFrame):
        return result

    return pd.DataFrame(result)


def write_csv(
    runs: list[dict[str, Any]], scores_df: pd.DataFrame, output_csv: Path
) -> pd.DataFrame:
    if not runs or len(scores_df) != len(runs):
        raise ValueError("Evaluation scores do not align with test cases")
    required_metrics = {"context_precision", "faithfulness", "answer_relevancy"}
    if not required_metrics.issubset(scores_df.columns):
        raise ValueError("Evaluation result is missing metrics")
    base_df = pd.DataFrame(runs)

    # Ensure indices are clean and aligned
    base_df = base_df.reset_index(drop=True)
    scores_df = scores_df.reset_index(drop=True)

    # Remove any duplicate columns from scores_df
    scores_df = scores_df.loc[:, ~scores_df.columns.duplicated(keep="first")]

    # Only keep metric columns from scores_df
    metric_columns = [
        col
        for col in scores_df.columns
        if col in {"context_precision", "faithfulness", "answer_relevancy"}
    ]
    scores_df = scores_df[metric_columns]
    scores_df = scores_df.apply(pd.to_numeric, errors="coerce")
    scores_df = scores_df.where((scores_df >= 0) & (scores_df <= 1))

    # Combine horizontally
    combined = pd.concat([base_df, scores_df], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # Build summary row
    summary_row = {}
    for col in combined.columns:
        if col in metric_columns:
            summary_row[col] = combined[col].mean()
        elif col == "case_id":
            summary_row[col] = "summary"
        elif col == "question":
            summary_row[col] = "AVERAGE"
        else:
            summary_row[col] = ""

    # Create summary dataframe ensuring same columns as combined
    summary_df = pd.DataFrame([summary_row], columns=combined.columns)

    # Append summary row
    final_df = pd.concat([combined, summary_df], axis=0, ignore_index=True)
    # Escape spreadsheet formulas only in the exported artifact, not scoring data.
    export_df = final_df.map(
        lambda value: (
            "'" + value
            if isinstance(value, str)
            and value.lstrip().startswith(("=", "+", "-", "@"))
            else value
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output_csv.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            export_df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_csv)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return final_df


def run_medrag_evaluation(
    *,
    question_runner: Callable[[str], dict[str, Any]],
    output_csv: Path | None = None,
) -> dict[str, Any]:
    load_env()
    output_path = Path(output_csv) if output_csv is not None else DEFAULT_OUTPUT_CSV

    runs = build_runs(question_runner)
    dataset = build_ragas_dataset(runs)
    scores_df = evaluate_with_ragas(dataset)
    final_df = write_csv(runs, scores_df, output_path)

    summary_row = final_df[final_df["case_id"] == "summary"].iloc[0].to_dict()
    case_rows = final_df[final_df["case_id"] != "summary"].to_dict(orient="records")

    return json_safe(
        {
            "provenance": {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "ragas_version": version("ragas"),
                "test_cases_sha256": hashlib.sha256(
                    json.dumps(TEST_CASES, sort_keys=True).encode()
                ).hexdigest(),
                "retrieved_contexts_sha256": hashlib.sha256(
                    json.dumps(
                        [run["retrieved_contexts"] for run in runs], sort_keys=True
                    ).encode()
                ).hexdigest(),
                "case_count": len(runs),
            },
            "summary": {
                "context_precision": summary_row.get("context_precision"),
                "faithfulness": summary_row.get("faithfulness"),
                "answer_relevancy": summary_row.get("answer_relevancy"),
            },
            "metric_valid_counts": {
                name: int(final_df.iloc[:-1][name].notna().sum())
                for name in ("context_precision", "faithfulness", "answer_relevancy")
            },
            "cases": case_rows,
            "output_csv": str(output_path.resolve()),
        }
    )


def json_safe(value):
    """Unavailable scores remain null, never a misleading zero or invalid JSON NaN."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
