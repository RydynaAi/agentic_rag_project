from langchain_core.documents import Document

from app import agents


def make_docs():
    return [
        Document(page_content="first passage", metadata={"source": "data/paper.pdf", "page": 0}),
        Document(page_content="second   passage\nwith breaks", metadata={"source": "data/paper.pdf", "page": 4}),
    ]


def test_build_source_uses_one_based_page_and_file_name():
    source = agents.build_source(make_docs()[1])
    assert source == {"source": "paper.pdf", "page": 5, "snippet": "second passage with breaks"}


def test_build_source_keeps_web_url():
    doc = Document(page_content="web text", metadata={"source": "https://example.com/a/b"})
    source = agents.build_source(doc)
    assert source["source"] == "https://example.com/a/b"
    assert source["page"] is None


def test_cited_passages_are_selected():
    docs = make_docs()
    assert agents.select_cited("The answer is here [2].", docs) == [docs[1]]


def test_uncited_answer_falls_back_to_all_passages():
    docs = make_docs()
    assert agents.select_cited("No citation here.", docs) == docs


def test_out_of_range_citation_is_ignored():
    docs = make_docs()
    assert agents.select_cited("See [7].", docs) == docs


def test_abstain_returns_message_and_marks_unsupported():
    result = agents.abstain({"retries": 3})
    assert result["generation"] == agents.ABSTAIN_MESSAGE
    assert result["supported"] is False
    assert result["sources"] == []


def test_fullwidth_brackets_are_recognised():
    docs = make_docs()
    assert agents.select_cited("The answer is here \u30102\u3011.", docs) == [docs[1]]
