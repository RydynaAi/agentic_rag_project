import os
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from app.llm import llm
from app.logger import logger
from app.retriever import retriever
from app.state import AgentState

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_GENERATIONS = 3
MIN_RELEVANT_DOCS = 1
SNIPPET_LENGTH = 200
ABSTAIN_MESSAGE = (
    "I could not find enough support in the available documents to answer this question reliably."
)

VECTORSTORE_DESCRIPTION = (
    "a single research article, Randomized Trial of a Generative AI Chatbot for Mental Health Treatment "
    "(NEJM AI, 2025), which evaluates Therabot, a fine-tuned generative AI therapy chatbot, "
    "in adults with depression, anxiety, or eating disorder risk"
)

ROUTER_INSTRUCTIONS = (
    "You route questions to one of two sources. "
    "The vectorstore contains " + VECTORSTORE_DESCRIPTION + ". "
    "Choose vectorstore for any question about this article, including its design, participants, methods, "
    "models, measures, statistics, results, and discussion, even if the question does not name the article. "
    "Choose web_search for everything else, such as current events, recently changing facts, who currently "
    "holds a role, prices, weather, software versions, or any topic outside the article. "
    "Return JSON only with key datasource, value either vectorstore or web_search."
)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_INSTRUCTIONS),
    ("human", "{question}")
])

router_chain = router_prompt | llm | JsonOutputParser()


def route_question(state: AgentState):
    result = router_chain.invoke({"question": state["question"]})
    route = "web_search" if result["datasource"] == "web_search" else "retrieve"
    logger.info("router | routed to %s", route)
    return route


def retrieve(state: AgentState):
    docs = retriever.invoke(state["question"])
    logger.info("retrieve | %d documents", len(docs))
    return {"documents": docs, "question": state["question"]}


grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess whether the following document is relevant to the question. Return JSON only with key score, value either yes or no."),
    ("human", "Document: {document}\n\nQuestion: {question}")
])

grade_chain = grade_prompt | llm | JsonOutputParser()


def grade_documents(state: AgentState):
    filtered = []

    for doc in state["documents"]:
        result = grade_chain.invoke({"document": doc.page_content, "question": state["question"]})
        if result["score"] == "yes":
            filtered.append(doc)

    web_search_needed = len(filtered) < MIN_RELEVANT_DOCS
    logger.info("grade_documents | kept %d, web_search_needed=%s", len(filtered), web_search_needed)
    return {"documents": filtered, "web_search": web_search_needed, "question": state["question"]}


rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question to be clearer and better optimized for semantic search. Return only the rewritten question."),
    ("human", "{question}")
])

rewrite_chain = rewrite_prompt | llm


def transform_query(state: AgentState):
    better_question = rewrite_chain.invoke({"question": state["question"]}).content
    return {"question": better_question, "documents": state["documents"]}


def web_search(state: AgentState):
    try:
        results = tavily.search(query=state["question"], max_results=3)
        web_docs = [
            Document(page_content=r["content"], metadata={"source": r.get("url", "web")})
            for r in results["results"]
        ]
    except Exception as exc:
        logger.warning("web_search | Tavily failed, continuing with internal documents: %s", exc)
        web_docs = []
    logger.info("web_search | %d web documents", len(web_docs))
    documents = state.get("documents", []) + web_docs
    return {"documents": documents, "question": state["question"]}


gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the numbered context passages. Cite the passages you used with their numbers in square brackets, for example [1]. If there is not enough information, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

gen_chain = gen_prompt | llm


def build_source(doc):
    metadata = doc.metadata or {}
    source = str(metadata.get("source", "unknown"))
    if not source.startswith("http"):
        source = os.path.basename(source)
    page = metadata.get("page")
    return {
        "source": source,
        "page": page + 1 if isinstance(page, int) else None,
        "snippet": " ".join(doc.page_content.split())[:SNIPPET_LENGTH],
    }


def select_cited(generation, documents):
    cited = {int(n) for n in re.findall(r"[\[\u3010](\d+)[\]\u3011]", generation)}
    chosen = [documents[i - 1] for i in sorted(cited) if 1 <= i <= len(documents)]
    return chosen or documents


def generate(state: AgentState):
    documents = state["documents"]
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(documents, 1))
    generation = gen_chain.invoke({"context": context, "question": state["question"]}).content
    generation = generation.replace("\u3010", "[").replace("\u3011", "]")
    attempt = state.get("retries", 0) + 1
    logger.info("generate | attempt %d", attempt)
    sources = [build_source(d) for d in select_cited(generation, documents)]
    return {
        "generation": generation,
        "documents": documents,
        "question": state["question"],
        "retries": attempt,
        "sources": sources,
        "supported": True,
    }


hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the answer fully supported by the given context? Return JSON only with key score, value either yes or no. This check reduces hallucination risk but does not guarantee correctness."),
    ("human", "Context: {documents}\n\nAnswer: {generation}")
])

hallucination_chain = hallucination_prompt | llm | JsonOutputParser()


def grade_generation(state: AgentState):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    result = hallucination_chain.invoke({"documents": context, "generation": state["generation"]})
    logger.info("grade_generation | supported=%s", result["score"])
    if result["score"] == "yes":
        return "useful"
    if state.get("retries", 0) >= MAX_GENERATIONS:
        return "max_retries"
    return "not supported"


def abstain(state: AgentState):
    logger.warning("abstain | answer not supported after %d attempts", state.get("retries", 0))
    return {"generation": ABSTAIN_MESSAGE, "sources": [], "supported": False}
