from typing import List
from typing_extensions import TypedDict


class AgentState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[str]
    retries: int
