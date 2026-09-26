# Agentic RAG System

A production-style Retrieval-Augmented Generation system that goes beyond standard RAG by adding dynamic routing, self-corrective retrieval grading (CRAG), query rewriting, and hallucination checking — orchestrated as a stateful graph with LangGraph.

## Overview

Instead of a linear "retrieve then generate" pipeline, this system treats question answering as a decision process. It routes each question to the right source, grades whether retrieved context is actually relevant, rewrites the question and falls back to live web search when local knowledge is insufficient, and checks its own generated answer against the source context before returning it.

## Architecture

The workflow is a directed graph with the following nodes:

- **Router** — classifies the question as answerable from the local knowledge base or requiring a web search
- **Retriever** — fetches the top relevant chunks from a ChromaDB vector store
- **Grading Agent (CRAG)** — evaluates each retrieved chunk's relevance to the question
- **Query Rewriter** — reformulates the question when retrieved context is insufficient
- **Web Search Agent** — falls back to real-time web search (Tavily) when local knowledge is inadequate
- **Generator** — produces an answer grounded strictly in the retrieved context
- **Hallucination Checker** — evaluates whether the generated answer is supported by the context and triggers regeneration if not. This reduces hallucination risk; it does not guarantee correctness, since it is one LLM judging another LLM's output.

Question → Router → [Retrieve → Grade → (Rewrite → Web Search) → Generate → Check] → Answer

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph / LangChain |
| LLM Inference | Groq API (GPT-OSS 120B) |
| Vector Store | ChromaDB |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5) |
| Web Search | Tavily Search API |
| API | FastAPI |
| UI | Streamlit |

## Project Structure
agentic_rag_project/
├── app/
│ ├── state.py # Shared graph state definition
│ ├── llm.py # LLM and embeddings setup
│ ├── retriever.py # Vector store and retriever
│ ├── agents.py # All agent nodes and prompts
│ └── graph.py # LangGraph workflow definition
├── data/ # Source PDF documents
├── ingest.py # Document ingestion pipeline
├── api.py # FastAPI service
├── streamlit_app.py # Streamlit demo UI
├── requirements.txt
└── .env.example

## Setup

1. Clone the repository and create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API keys:

GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

4. Add PDF documents to the `data/` folder, then build the vector store:
```bash
python ingest.py
```

5. Run the API:
```bash
uvicorn api:api --reload
```

6. In a separate terminal, run the UI:
```bash
streamlit run streamlit_app.py
```

## Example

**Question:** How was participant identity verified in the study?

**Answer:** Participant identity was confirmed through a video-verification step. Before the study began, each potential respondent took part in a brief pre-study video call in which the researcher could compare the participant's appearance and any presented ID against their screener responses...

## Design Notes

- Retrieval grading (CRAG) prevents the generator from answering off irrelevant chunks, rather than relying on retrieval similarity scores alone.
- The web search fallback only activates when local documents are graded as insufficient, keeping answers grounded in the local knowledge base when possible.
- The hallucination check is a best-effort self-verification step, not a formal guarantee of factual correctness.