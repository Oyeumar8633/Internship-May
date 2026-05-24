# Task 4 — Context-Aware RAG Chatbot (LangChain + FAISS)

**DevelopersHub Corporation — Internship Task 4**

Conversational **RAG** chatbot with **FAISS**, **Hugging Face** embeddings, **Groq** LLM, and **Streamlit** in Colab.

---

## Folder structure

```
Task 4/
├── README.md
├── rag_chatbot.ipynb      # Run Cells 1–5 only (nothing else to paste)
├── app.py                 # Also written by notebook Cell 4
├── knowledge_base/        # Created by Cell 2
└── faiss_devhub_index/    # Created by Cell 2
```

---

## Quick start (Google Colab)

1. Colab **Secrets** → add `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)).
2. Upload `rag_chatbot.ipynb`.
3. **Runtime → Run all** (or Cells **1 → 5** in order).
4. Cell **5** opens the chat in a **new browser tab**.
5. Click **Start chatbot**, then ask e.g. *How many PTO days?*

Do **not** paste extra deploy cells from chat. Do **not** use localtunnel (`loca.lt`) — it breaks Streamlit on Colab.

---

## Notebook cells

| Cell | What it does |
|------|----------------|
| **1** | Install packages + load `GROQ_API_KEY` from Secrets |
| **2** | Build KB, chunk (500/50), FAISS index |
| **3** | Test `ConversationalRetrievalChain` + memory (two questions) |
| **4** | Write `app.py` |
| **5** | Start Streamlit, wait until port 8501 is up, open UI in new tab |

If `faiss_devhub_index` already exists and you only changed the UI, rerun **Cells 1, 4, 5**.

---

## Architecture

```
User → Streamlit → ConversationalRetrievalChain
                        ├─ FAISS (all-MiniLM-L6-v2)
                        └─ Groq llama-3.1-8b-instant
```

---

## Knowledge base topics

Company overview, PTO, remote work, security, MLOps platform, internship program, support SLA, API guidelines.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY missing` | Add secret in Colab; rerun Cell 1 or 5 |
| `connection refused` on 8501 | Rerun Cell 5; read crash log printed in cell output |
| `No module named langchain.chains` | Rerun Cell 1 (`langchain-classic` installed there) |
| Embeddings / torch errors | Runtime → **Restart session**, then Cells 1–2 (do not `pip install torch` manually) |
