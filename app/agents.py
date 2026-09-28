import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from tavily import TavilyClient
from app.state import AgentState
from app.llm import llm
from app.retriever import retriever

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_GENERATIONS = 3

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the question as vectorstore or web_search. Return JSON only with key datasource, value either vectorstore or web_search."),
    ("human", "{question}")
])

router_chain = router_prompt | llm | JsonOutputParser()

def route_question(state: AgentState):
    result = router_chain.invoke({"question": state["question"]})
    return "web_search" if result["datasource"] == "web_search" else "retrieve"

def retrieve(state: AgentState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess whether the following document is relevant to the question. Return JSON only with key score, value either yes or no."),
    ("human", "Document: {document}\n\nQuestion: {question}")
])

grade_chain = grade_prompt | llm | JsonOutputParser()

def grade_documents(state: AgentState):
    filtered = []
    web_search_needed = False

    for doc in state["documents"]:
        result = grade_chain.invoke({"document": doc.page_content, "question": state["question"]})
        if result["score"] == "yes":
            filtered.append(doc)
        else:
            web_search_needed = True

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
    results = tavily.search(query=state["question"], max_results=3)
    web_docs = [Document(page_content=r["content"]) for r in results["results"]]
    documents = state.get("documents", []) + web_docs
    return {"documents": documents, "question": state["question"]}

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the given context. If there is not enough information, say you don't know."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

gen_chain = gen_prompt | llm

def generate(state: AgentState):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    generation = gen_chain.invoke({"context": context, "question": state["question"]}).content
    return {"generation": generation, "documents": state["documents"], "question": state["question"], "retries": state.get("retries", 0) + 1}

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the answer fully supported by the given context? Return JSON only with key score, value either yes or no. This check reduces hallucination risk but does not guarantee correctness."),
    ("human", "Context: {documents}\n\nAnswer: {generation}")
])

hallucination_chain = hallucination_prompt | llm | JsonOutputParser()

def grade_generation(state: AgentState):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    result = hallucination_chain.invoke({"documents": context, "generation": state["generation"]})
    if result["score"] == "yes":
        return "useful"
    if state.get("retries", 0) >= MAX_GENERATIONS:
        return "max_retries"
    return "not supported"
