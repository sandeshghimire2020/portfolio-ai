import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8001/chat"

st.set_page_config(page_title="Portfolio AI — Test Chat")
st.title("Portfolio AI — RAG Chat")
st.caption("Powered by ChromaDB + OpenAI API")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about Sandesh..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    API_URL,
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=30,
                )
                answer = res.json().get("response", "No response from API.")
            except Exception as e:
                answer = f"Error connecting to API: {e}\n\nMake sure `uvicorn api:app --reload --port 8001` is running."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
