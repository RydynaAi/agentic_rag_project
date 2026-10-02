from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import agents
from app.graph import app as graph


class FakeChain:
    def __init__(self, results):
        self.results = list(results)

    def invoke(self, _):
        return self.results.pop(0)


class FakeRetriever:
    def invoke(self, _):
        return [Document(page_content="internal doc")]


class FakeTavily:
    def __init__(self):
        self.calls = 0

    def search(self, query, max_results):
        self.calls += 1
        return {"results": [{"content": "web doc"}]}


def setup(monkeypatch, route, grades, checks):
    tavily = FakeTavily()
    monkeypatch.setattr(agents, "router_chain", FakeChain([{"datasource": route}]))
    monkeypatch.setattr(agents, "retriever", FakeRetriever())
    monkeypatch.setattr(agents, "grade_chain", FakeChain(grades))
    monkeypatch.setattr(agents, "rewrite_chain", FakeChain([AIMessage(content="better question")]))
    monkeypatch.setattr(agents, "gen_chain", FakeChain([AIMessage(content="final answer")] * 3))
    monkeypatch.setattr(agents, "hallucination_chain", FakeChain(checks))
    monkeypatch.setattr(agents, "tavily", tavily)
    return tavily


def test_relevant_docs_skip_web_search(monkeypatch):
    tavily = setup(monkeypatch, "vectorstore", [{"score": "yes"}], [{"score": "yes"}])
    result = graph.invoke({"question": "q"})
    assert result["generation"] == "final answer"
    assert tavily.calls == 0


def test_irrelevant_docs_trigger_rewrite_and_web_search(monkeypatch):
    tavily = setup(monkeypatch, "vectorstore", [{"score": "no"}], [{"score": "yes"}])
    result = graph.invoke({"question": "q"})
    assert result["question"] == "better question"
    assert result["generation"] == "final answer"
    assert tavily.calls == 1


def test_router_can_go_directly_to_web_search(monkeypatch):
    tavily = setup(monkeypatch, "web_search", [], [{"score": "yes"}])
    result = graph.invoke({"question": "q"})
    assert result["generation"] == "final answer"
    assert tavily.calls == 1


def test_unsupported_answer_is_regenerated_once(monkeypatch):
    setup(monkeypatch, "vectorstore", [{"score": "yes"}], [{"score": "no"}, {"score": "yes"}])
    result = graph.invoke({"question": "q"})
    assert result["generation"] == "final answer"


def test_generation_stops_after_max_retries(monkeypatch):
    setup(monkeypatch, "vectorstore", [{"score": "yes"}], [{"score": "no"}] * 5)
    result = graph.invoke({"question": "q"})
    assert result["generation"] == agents.ABSTAIN_MESSAGE
    assert result["supported"] is False
    assert result["sources"] == []
    assert result["retries"] == agents.MAX_GENERATIONS


class FailingTavily:
    def search(self, query, max_results):
        raise RuntimeError("tavily down")


def test_web_search_failure_falls_back_to_internal_docs(monkeypatch):
    setup(monkeypatch, "vectorstore", [{"score": "no"}], [{"score": "yes"}])
    monkeypatch.setattr(agents, "tavily", FailingTavily())
    result = graph.invoke({"question": "q"})
    assert result["generation"] == "final answer"
