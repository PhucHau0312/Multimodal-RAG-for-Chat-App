import json
import os
import streamlit as st

from src.rag import RAGChain


UPLOAD_FOLDER = os.getenv("UPLOAD_DIR")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HISTORY_DIR = os.getenv("HISTORY_DIR")
os.makedirs(HISTORY_DIR, exist_ok=True)


def save_history(session_id, messages):
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(messages, f)

def load_history(session_id):
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":

    chain = RAGChain()
    
    session_id = st.text_input("Enter Session ID to load history:", value="default_session")

    if st.button("Load Chat"):
        if os.path.exists( os.path.join(HISTORY_DIR, f"{session_id}.json")):
            st.session_state.messages = load_history(session_id)
        else: 
            st.session_state.messages = []

    with st.sidebar:
        st.header("📂 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDFs to update RAG", 
            type="pdf", 
            accept_multiple_files=True
        )

        if st.button("Update Knowledge"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        chain.update_knowledge(save_path)
                        st.success(f"Added {uploaded_file.name}")
            else:
                st.error("Please select a file first.")

    st.title("💬 Chat App")

    if len(st.session_state.messages) != 0:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):        
            response = st.write_stream(chain.run(prompt, st.session_state.messages[-6:])["answer"])
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_history(session_id, st.session_state.messages)