from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import app as rag_app

api = FastAPI()

class Query(BaseModel):
    question: str

@api.post("/ask")
def ask(query: Query):
    result = rag_app.invoke({"question": query.question})
    return {"answer": result["generation"]}