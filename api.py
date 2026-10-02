from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from app.graph import app as rag_app
from app.logger import logger

api = FastAPI()


class Query(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@api.post("/ask")
def ask(query: Query):
    try:
        result = rag_app.invoke({"question": query.question})
    except Exception:
        logger.exception("ask | pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="The pipeline failed while answering your question. Please try again.",
        )
    return {
        "answer": result["generation"],
        "sources": result.get("sources", []),
        "supported": result.get("supported", True),
    }
