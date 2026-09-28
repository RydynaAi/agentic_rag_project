from app import agents


class FakeChain:
    def __init__(self, result):
        self.result = result

    def invoke(self, _):
        return self.result


def test_routes_to_web_search(monkeypatch):
    monkeypatch.setattr(agents, "router_chain", FakeChain({"datasource": "web_search"}))
    assert agents.route_question({"question": "latest news"}) == "web_search"


def test_routes_to_retrieve(monkeypatch):
    monkeypatch.setattr(agents, "router_chain", FakeChain({"datasource": "vectorstore"}))
    assert agents.route_question({"question": "what is attention"}) == "retrieve"
