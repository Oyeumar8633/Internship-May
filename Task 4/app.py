# Loads chat ONLY after you click Start — avoids crash on page open

import streamlit as st

st.set_page_config(page_title="DevelopersHub RAG Assistant", page_icon="💬")
st.title("DevelopersHub RAG Assistant")
st.caption("Click **Start chatbot** once, then ask questions.")

if "history" not in st.session_state:
    st.session_state.history = []
if "chain" not in st.session_state:
    st.session_state.chain = None


def build_chain():
    import json
    import os
    from pathlib import Path

    from langchain_community.vectorstores import FAISS

    try:
        from langchain_classic.chains import ConversationalRetrievalChain
    except ImportError:
        from langchain.chains import ConversationalRetrievalChain

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY missing. Run the DEPLOY cell in Colab first.")

    if not Path("faiss_devhub_index").exists():
        raise FileNotFoundError("Missing faiss_devhub_index — run notebook Cell 2.")

    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local("faiss_devhub_index", emb, allow_dangerous_deserialization=True)

    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=key)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )


if st.session_state.chain is None:
    if st.button("Start chatbot", type="primary"):
        with st.spinner("Loading FAISS + Groq (~1 min first time)…"):
            try:
                st.session_state.chain = build_chain()
                st.success("Ready! Groq is connected. Ask a question below.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    st.stop()

q = st.chat_input("Ask about DevelopersHub (e.g. How many PTO days?)")
if q:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = st.session_state.chain.invoke(
                    {"question": q, "chat_history": st.session_state.history}
                )
                ans = r.get("answer", "")
                docs = r.get("source_documents", [])
            except Exception as e:
                ans = f"Error: {e}"
                docs = []
        st.markdown(ans)
        if docs:
            with st.expander("Sources"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {d.page_content[:200]}…")
    st.session_state.history.append((q, ans))
