from langchain_core.documents import Document

from app import agents


class FakeChain:
    def __init__(self, results):
        self.results = list(results)

    def invoke(self, _):
        return self.results.pop(0)


def test_relevant_docs_are_kept_and_no_web_search(monkeypatch):
    monkeypatch.setattr(agents, "grade_chain", FakeChain([{"score": "yes"}, {"score": "yes"}]))
    state = {"question": "q", "documents": [Document(page_content="a"), Document(page_content="b")]}
    result = agents.grade_documents(state)
    assert len(result["documents"]) == 2
    assert result["web_search"] is False


def test_irrelevant_doc_is_dropped_and_triggers_web_search(monkeypatch):
    monkeypatch.setattr(agents, "grade_chain", FakeChain([{"score": "yes"}, {"score": "no"}]))
    state = {"question": "q", "documents": [Document(page_content="a"), Document(page_content="b")]}
    result = agents.grade_documents(state)
    assert [d.page_content for d in result["documents"]] == ["a"]
    assert result["web_search"] is True


def test_supported_answer_is_useful(monkeypatch):
    monkeypatch.setattr(agents, "hallucination_chain", FakeChain([{"score": "yes"}]))
    state = {"documents": [Document(page_content="a")], "generation": "answer"}
    assert agents.grade_generation(state) == "useful"


def test_unsupported_answer_is_flagged(monkeypatch):
    monkeypatch.setattr(agents, "hallucination_chain", FakeChain([{"score": "no"}]))
    state = {"documents": [Document(page_content="a")], "generation": "answer"}
    assert agents.grade_generation(state) == "not supported"
