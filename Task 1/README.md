# Task 1 — News Topic Classifier (BERT + AG News)

**DevelopersHub Corporation — Internship Task 1**  
**Author:** Muhammad Umar Iftikhar

Fine-tune `bert-base-uncased` on the [AG News](https://huggingface.co/datasets/ag_news) dataset and deploy an interactive **Streamlit** demo from **Google Colab**, exposed via a public tunnel.

---

## Overview

This project implements a production-style NLP pipeline:

1. Load and tokenize AG News (4 topic classes)
2. Fine-tune BERT for sequence classification
3. Evaluate with **Accuracy** and **Weighted F1**
4. Save the model locally
5. Serve predictions through a Streamlit web UI with a shareable public URL

---

## Folder structure

```
Task 1/
├── README.md                      # This file
├── news_topic_classifier.ipynb    # End-to-end Colab notebook (train + deploy)
└── app.py                         # Streamlit frontend (loads ./saved_bert_model/)
```

### Generated at runtime (Colab)

After training (Notebook **Cell 4**), these are created in the Colab working directory:

```
./saved_bert_model/                # Fine-tuned weights + tokenizer
./bert_ag_news_checkpoints/        # Training checkpoints (optional)
```

> `saved_bert_model/` is not committed to git by default (large files). It exists on Colab after training completes.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Dataset | Hugging Face `datasets` — `ag_news` |
| Model | `bert-base-uncased` via `transformers` |
| Training | Hugging Face `Trainer` + `TrainingArguments` |
| Metrics | scikit-learn (accuracy, weighted F1) |
| UI | Streamlit |
| Public URL | localtunnel (default) or ngrok (optional) |
| Runtime | Google Colab (GPU recommended) |

---

## Topic labels

| Hugging Face ID | Assignment ID | Topic |
|-----------------|---------------|--------|
| 0 | 1 | World |
| 1 | 2 | Sports |
| 2 | 3 | Business |
| 3 | 4 | Sci/Tech |

The model predicts IDs **0–3**. The Streamlit app shows the assignment mapping (1–4) in the UI caption.

---

## How to run (Google Colab)

### 1. Open the notebook

Upload `news_topic_classifier.ipynb` to [Google Colab](https://colab.research.google.com/) or open from Drive.

### 2. Enable GPU

**Runtime → Change runtime type → T4 GPU** (required for training; ~30–45 min).

### 3. Run cells in order

| Cell | Purpose |
|------|---------|
| **1** | Install dependencies (`transformers`, `datasets`, `streamlit`, etc.) |
| **2** | Load AG News, tokenize with BERT (max length 128) |
| **3** | Load model, set training args, define metrics |
| **4** | Train, evaluate, save to `./saved_bert_model/` |
| **5** | Write `app.py` to disk |
| **6** | Start Streamlit + public tunnel |

### 4. Already trained?

If `./saved_bert_model/` and `app.py` exist, **do not rerun Cells 1–4**. Run only:

- **Cell 5** (if `app.py` is missing)
- **Cell 6** (or the deploy snippet below)

---

## Deploy only (after training)

Use this if training finished and you only need the live demo:

```python
# 1) Ensure app.py is current (run Cell 5 once)
# 2) Run deploy cell — uses localtunnel (no ngrok account required)

import os, re, subprocess, sys, time
from pathlib import Path

PORT = 8501
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "streamlit"], check=False)
subprocess.run(["pkill", "-f", "streamlit"], check=False)
time.sleep(2)

env = os.environ.copy()
env["STREAMLIT_SERVER_HEADLESS"] = "true"
subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py",
     "--server.port", str(PORT), "--server.address", "0.0.0.0",
     "--server.enableCORS", "false", "--server.enableXsrfProtection", "false",
     "--server.fileWatcherType", "none"],
    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(8)

proc = subprocess.Popen(
    ["npx", "--yes", "localtunnel", "--port", str(PORT)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for _ in range(90):
    line = proc.stdout.readline()
    if line:
        print(line.rstrip())
        m = re.search(r"https://[a-zA-Z0-9-]+\.loca\.lt", line)
        if m:
            print("\nPUBLIC URL:", m.group(0))
            break
```

Open the printed `https://….loca.lt` URL. Click **Continue** if localtunnel prompts once. Keep the cell running while testing.

---

## Training configuration (defaults)

| Parameter | Value |
|-----------|--------|
| Base model | `bert-base-uncased` |
| Max sequence length | 128 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Mixed precision | `fp16=True` when CUDA is available |

---

## Streamlit app (`app.py`)

- Loads the fine-tuned model from `./saved_bert_model/`
- Uses `st.session_state` for one-time model load (Colab-safe; no `@st.cache_resource`)
- Text input + **Classify** button
- Shows predicted topic, confidence %, and per-class probabilities

### Run locally (optional)

If you have copied `saved_bert_model/` to your machine:

```bash
pip install streamlit torch transformers
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `torchvision::nms` / import errors after Cell 1 | Do **not** reinstall `torch` on Colab. Restart runtime; rerun from Cell 2. |
| `Trainer` unexpected keyword `tokenizer` | Notebook uses `processing_class` with fallback — use latest `news_topic_classifier.ipynb`. |
| `model` not defined in Cell 4 | Rerun Cells 2 → 3 → 4 in the **same session** (no runtime restart between them). |
| ngrok `ERR_NGROK_4018` | ngrok needs a free authtoken, or use **localtunnel** (deploy snippet above). |
| `TypeError: Importing a module script failed` | Use the provided `app.py` (no `@st.cache_resource`, no `if __name__ == "__main__"`). |
| `cannot import AutoModelForSequenceClassification` | Run `!pip install -U transformers`, restart runtime, redeploy only. |
| Tunnel works but app errors | Rerun Cell 5 to refresh `app.py`, then redeploy. |

---

## Deliverables checklist

- [x] Fine-tuned BERT on AG News (4 classes)
- [x] Evaluation: Accuracy + Weighted F1 + classification report
- [x] Saved model: `./saved_bert_model/`
- [x] Streamlit UI: `app.py`
- [x] Live public demo URL via tunnel from Colab

---

## References

- [AG News dataset](https://huggingface.co/datasets/ag_news)
- [BERT base uncased](https://huggingface.co/bert-base-uncased)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit](https://streamlit.io/)
