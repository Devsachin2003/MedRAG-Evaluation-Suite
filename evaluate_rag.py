"""Automated MedRAG evaluation script using ragas.

This script:
- queries the existing FastAPI `/chat` endpoint
- captures generated answers plus retrieved chunks
- evaluates Context Precision, Answer Faithfulness, and Answer Relevancy
- writes detailed results and summary scores to CSV
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.eval_core import DEFAULT_OUTPUT_CSV, query_chat_api, run_medrag_evaluation  # noqa: E402


DEFAULT_API_URL = os.getenv("MEDRAG_API_URL", "http://127.0.0.1:8000")


def configure_logging() -> None:
    """Configure the script logger for command-line evaluation runs."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def main() -> None:
    """Run the MedRAG evaluation workflow against the configured backend."""

    parser = argparse.ArgumentParser(description="Run MedRAG evaluation with ragas.")
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL, help="Base URL for the FastAPI backend."
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to the CSV file that will contain detailed scores and summary metrics.",
    )
    args = parser.parse_args()

    configure_logging()
    result = run_medrag_evaluation(
        question_runner=lambda question: query_chat_api(args.api_url, question),
        output_csv=Path(args.output_csv),
    )

    summary = result["summary"]
    logging.info("Saved evaluation results to %s", result["output_csv"])
    logging.info(
        "Context Precision: %s | Answer Faithfulness: %s | Answer Relevancy: %s",
        summary.get("context_precision"),
        summary.get("faithfulness"),
        summary.get("answer_relevancy"),
    )


if __name__ == "__main__":
    main()
