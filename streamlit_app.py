import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Agentic RAG Assistant")
question = st.text_input("Ask me anything")

if st.button("Send"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=120,
                )
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Make sure it is running: uvicorn api:api --reload")
        except requests.exceptions.Timeout:
            st.error("The request took too long. Please try again.")
        else:
            if response.ok:
                st.write(response.json()["answer"])
            elif response.status_code == 422:
                st.warning("Please enter a valid question.")
            else:
                st.error("Something went wrong while answering. Please try again.")
