import json

import pandas as pd
import pytest

from app import eval_core, rag_core, settings


def test_production_requires_credentials(monkeypatch):
    monkeypatch.setattr(settings, "PRODUCTION", True)
    monkeypatch.setattr(settings, "API_KEY", "")
    with pytest.raises(RuntimeError, match="MEDRAG_API_KEY"):
        settings.validate_production()


def test_no_context_never_calls_provider():
    assert "Insufficient evidence" in rag_core.generate_grounded_response(
        "medical question", []
    )


def test_pdf_signature():
    with pytest.raises(ValueError):
        rag_core.read_upload(b"not pdf", "test.pdf")


def test_raw_retention_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("RETAIN_UPLOADS", raising=False)
    monkeypatch.setattr(rag_core, "UPLOAD_DIR", tmp_path)
    assert rag_core.persist_upload_copy(b"private", "sample.txt") is None
    assert not list(tmp_path.iterdir())


def test_nonfinite_scores_are_null():
    result = eval_core.json_safe({"score": float("nan"), "cases": [float("inf")]})
    assert json.dumps(result, allow_nan=False) == '{"score": null, "cases": [null]}'


def test_atomic_csv_and_formula_escape(tmp_path):
    metrics = pd.DataFrame(
        {
            key: [0.5]
            for key in ["faithfulness", "context_precision", "answer_relevancy"]
        }
    )
    path = tmp_path / "reports" / "output.csv"
    result = eval_core.write_csv(
        [{"case_id": "case", "question": "=1+1"}], metrics, path
    )
    assert result.iloc[0]["question"] == "=1+1"
    assert "'=1+1" in path.read_text()
    assert not list(path.parent.glob("*.tmp"))


def test_misaligned_scores_rejected(tmp_path):
    with pytest.raises(ValueError, match="align"):
        eval_core.write_csv([{"case_id": "test"}], pd.DataFrame(), tmp_path / "out.csv")


def test_invalid_scores_are_not_averaged(tmp_path):
    metrics = pd.DataFrame(
        {
            key: [float("inf"), -1, 0.6]
            for key in ["faithfulness", "context_precision", "answer_relevancy"]
        }
    )
    result = eval_core.write_csv(
        [{"case_id": str(i)} for i in range(3)], metrics, tmp_path / "scores.csv"
    )
    assert pd.isna(result.iloc[0]["faithfulness"])
    assert result.iloc[-1]["faithfulness"] == 0.6


def test_evaluation_pipeline_provenance(tmp_path, monkeypatch):
    def scores(dataset):
        assert len(dataset) == len(eval_core.TEST_CASES)
        return pd.DataFrame(
            {
                key: [0.75] * len(dataset)
                for key in ["faithfulness", "context_precision", "answer_relevancy"]
            }
        )

    monkeypatch.setattr(eval_core, "evaluate_with_ragas", scores)
    result = eval_core.run_medrag_evaluation(
        question_runner=lambda _: {
            "answer": "grounded",
            "sources": [],
            "retrieved_contexts": ["evidence"],
        },
        output_csv=tmp_path / "report.csv",
    )
    assert result["metric_valid_counts"]["faithfulness"] == 3
    assert result["summary"]["faithfulness"] == 0.75
    assert len(result["provenance"]["retrieved_contexts_sha256"]) == 64
    assert (tmp_path / "report.csv").exists()


def test_concurrent_ingest_is_idempotent(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import chromadb

    class Embeddings:
        def __call__(self, input):
            return [[1.0, 0.0, 0.0] for _ in input]

        def name(self):
            return "test"

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection(
        "concurrent", embedding_function=Embeddings()
    )
    monkeypatch.setattr(rag_core, "_collection", collection)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: rag_core.ingest_document(
                    b"one medical guideline", "test.txt"
                ),
                range(4),
            )
        )
    assert sum(r["chunks_stored"] for r in results) == 1
    assert collection.count() == 1
