from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
