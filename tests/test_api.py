from fastapi.testclient import TestClient

import api as api_module


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def invoke(self, _):
        if self.error:
            raise self.error
        return self.result


def make_client(monkeypatch, graph):
    monkeypatch.setattr(api_module, "rag_app", graph)
    return TestClient(api_module.api)


def test_ask_returns_answer(monkeypatch):
    client = make_client(monkeypatch, FakeGraph(result={"generation": "ok"}))
    response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "ok", "sources": [], "supported": True}


def test_empty_question_is_rejected(monkeypatch):
    client = make_client(monkeypatch, FakeGraph(result={"generation": "ok"}))
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_whitespace_question_is_rejected(monkeypatch):
    client = make_client(monkeypatch, FakeGraph(result={"generation": "ok"}))
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_pipeline_failure_returns_clear_error(monkeypatch):
    client = make_client(monkeypatch, FakeGraph(error=RuntimeError("boom")))
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 500
    assert "failed" in response.json()["detail"]


def test_sources_and_support_flag_are_returned(monkeypatch):
    sources = [{"source": "paper.pdf", "page": 3, "snippet": "text"}]
    graph = FakeGraph(result={"generation": "declined", "sources": sources, "supported": False})
    client = make_client(monkeypatch, graph)
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["sources"] == sources
    assert body["supported"] is False
