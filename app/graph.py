from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.agents import (
    route_question,
    retrieve,
    grade_documents,
    transform_query,
    web_search,
    generate,
    grade_generation
)

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.set_conditional_entry_point(
    route_question,
    {"web_search": "web_search", "retrieve": "retrieve"}
)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    lambda s: "transform_query" if s["web_search"] else "generate",
    {"transform_query": "transform_query", "generate": "generate"}
)

workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")

workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {"useful": END, "not supported": "generate"}
)

app = workflow.compile()
