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
                data = response.json()
                st.write(data["answer"])
                if not data.get("supported", True):
                    st.warning("The system could not verify an answer against the documents, so it declined to answer.")
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for source in sources:
                            page = f", page {source['page']}" if source.get("page") else ""
                            st.markdown(f"**{source['source']}{page}**")
                            st.caption(source["snippet"])
            elif response.status_code == 422:
                st.warning("Please enter a valid question.")
            else:
                st.error("Something went wrong while answering. Please try again.")
