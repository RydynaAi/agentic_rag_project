from app.config import validate_env
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

validate_env()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
