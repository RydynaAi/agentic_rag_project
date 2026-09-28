import os
import sys
import types

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")

fake_llm = types.ModuleType("app.llm")
fake_llm.llm = RunnableLambda(lambda _: AIMessage(content="{}"))

fake_retriever = types.ModuleType("app.retriever")
fake_retriever.retriever = RunnableLambda(lambda _: [])

sys.modules["app.llm"] = fake_llm
sys.modules["app.retriever"] = fake_retriever
