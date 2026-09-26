import streamlit as st
import requests

st.title("Agentic RAG Assistant")
question = st.text_input("Ask me anything")

if st.button("Send") and question:
    response = requests.post("http://localhost:8000/ask", json={"question": question})
    st.write(response.json()["answer"])