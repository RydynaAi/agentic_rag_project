from langchain_community.vectorstores import Chroma
from app.llm import embeddings

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
