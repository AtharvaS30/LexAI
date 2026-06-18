import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import build_vectorstore, query_rag

load_dotenv()

st.set_page_config(
    page_title="LexAI - Legal Document Q&A",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ LexAI")
st.subheader("Legal Document Q&A System")
st.markdown("Upload a legal PDF and ask questions about it in plain English.")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "filename" not in st.session_state:
    st.session_state.filename = None

# Sidebar - Upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a legal PDF", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.filename:
            with st.spinner("Processing document..."):
                # Save temp file
                import tempfile
                temp_path = os.path.join(
                    tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Build vectorstore
                vs = build_vectorstore(temp_path)
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.filename = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✅ Ready: {uploaded_file.name}")
                else:
                    st.error("Failed to process document.")

    if st.session_state.filename:
        st.info(f"Active: {st.session_state.filename}")
        if st.button("🗑️ Clear Document"):
            st.session_state.vectorstore = None
            st.session_state.filename = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("**Powered by**")
    st.markdown("- 🔍 FAISS Vector Search")
    st.markdown("- 🧠 Gemini Embeddings")
    st.markdown("- 🤖 Groq LLaMA 3")

# Chat area
if st.session_state.vectorstore is None:
    st.info("👈 Upload a legal PDF from the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_query = st.chat_input("Ask a question about the document...")

    if user_query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query})

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = query_rag(
                    user_query,
                    st.session_state.vectorstore,
                    st.session_state.chat_history[:-1]
                )
            st.markdown(answer)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer})
