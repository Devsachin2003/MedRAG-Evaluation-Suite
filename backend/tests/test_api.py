import pytest
from fastapi.testclient import TestClient

import main as api


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api.settings, "API_KEY", "")

    async def _noop():
        return None

    monkeypatch.setattr(api, "warmup_embedding_model", _noop)
    monkeypatch.setattr(api, "collection_count", lambda: 0)
    monkeypatch.setattr(
        api,
        "ingest_document",
        lambda content, filename: {
            "filename": filename,
            "chunks_stored": 1,
            "chunks_skipped": 0,
        },
    )
    monkeypatch.setattr(
        api, "persist_upload_copy", lambda content, filename: "stored.txt"
    )

    def _fake_eval(*, question_runner, output_csv):
        return {
            "summary": {
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
            },
            "cases": [],
            "output_csv": str(output_csv),
        }

    monkeypatch.setattr(api, "run_medrag_evaluation", _fake_eval)

    with TestClient(api.app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "collection_count" in payload


def test_version_endpoint(client, monkeypatch):
    monkeypatch.setenv("MEDRAG_VERSION", "1.2.3")
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": "1.2.3"}


def test_ingest_upload_happy_path(client):
    response = client.post(
        "/api/ingest/upload",
        files={"file": ("sample.txt", b"hello medrag", "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.txt"
    assert payload["chunks_stored"] == 1
    assert "stored_file" in payload


def test_run_evaluation_contract(client):
    response = client.post("/api/evaluation/run", json={})
    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert "cases" in payload
    assert "output_csv" in payload


@pytest.mark.integration
def test_ingest_integration(tmp_path, monkeypatch):
    import chromadb
    from app import rag_core

    class Embeddings:
        def __call__(self, input):
            return [[1.0, 0.0, 0.0] for _ in input]

        def name(self):
            return "test-embeddings"

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection(
        "integration", embedding_function=Embeddings()
    )
    monkeypatch.setattr(rag_core, "_collection", collection)

    content = b"This is a test document for integration."
    result = rag_core.ingest_document(content, "sample.txt")
    assert result["chunks_stored"] >= 1
    repeated = rag_core.ingest_document(content, "sample.txt")
    assert repeated["chunks_stored"] == 0
    assert collection.count() == 1


@pytest.mark.parametrize("query", ["", "   ", "x" * 8001])
def test_invalid_chat(client, query):
    response = client.post("/chat", json={"query": query})
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


@pytest.mark.parametrize(
    "filename", ["../../secret.csv", "/tmp/secret.csv", "bad.txt", "a\\b.csv"]
)
def test_export_path_rejected(client, filename):
    assert (
        client.post("/api/evaluation/run", json={"output_csv": filename}).status_code
        == 422
    )


def test_export_does_not_expose_server_path(client):
    result = client.post(
        "/api/evaluation/run", json={"output_csv": "report.csv"}
    ).json()
    assert "/" not in result["output_csv"]
    assert result["output_csv"].startswith("report-")


def test_authentication(client, monkeypatch):
    monkeypatch.setattr(api.settings, "API_KEY", "test-key")
    assert client.post("/api/evaluation/run", json={}).status_code == 401
    assert (
        client.post(
            "/api/evaluation/run", json={}, headers={"Authorization": "Bearer test-key"}
        ).status_code
        == 200
    )
    assert client.get("/live").status_code == 200


@pytest.mark.parametrize(
    "filename,status", [("../../test.txt", 400), ("test.exe", 415)]
)
def test_unsafe_uploads(client, filename, status):
    assert (
        client.post(
            "/api/ingest/upload", files={"file": (filename, b"test")}
        ).status_code
        == status
    )


def test_upload_limit(client, monkeypatch):
    monkeypatch.setattr(api.settings, "MAX_UPLOAD_BYTES", 4)
    assert (
        client.post(
            "/api/ingest/upload", files={"file": ("test.txt", b"12345")}
        ).status_code
        == 413
    )


def test_chunked_body_limit(client, monkeypatch):
    monkeypatch.setattr(api.settings, "MAX_UPLOAD_BYTES", 4)
    response = client.post("/chat", content=iter([b"x" * 40000, b"x" * 40000]))
    assert response.status_code == 413


def test_safe_failure(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("patient-private-secret")

    monkeypatch.setattr(api, "retrieve_context", fail)
    response = client.post("/chat", json={"query": "a question"})
    assert response.status_code == 500
    assert "patient-private-secret" not in response.text
    assert response.headers["X-Request-ID"]


def test_capacity(client):
    api.app.state.active = api.settings.MAX_CONCURRENT_REQUESTS
    assert client.post("/chat", json={"query": "question"}).status_code == 503


def test_evaluation_already_running(client):
    api.app.state.evaluation_running = True
    assert client.post("/api/evaluation/run", json={}).status_code == 409


def test_stream_failure_is_explicit(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve_context", lambda *args, **kw: [])

    async def stream(*args):
        yield "partial"
        raise RuntimeError("private")

    monkeypatch.setattr(api, "stream_grounded_response", stream)
    response = client.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "hello"}]}
    )
    assert '"type": "error"' in response.text
    assert '"type": "done"' not in response.text
    assert "private" not in response.text
