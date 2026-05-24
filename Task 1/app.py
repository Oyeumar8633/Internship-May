# Streamlit AG News classifier — no @st.cache_resource (fixes Colab "Importing a module script failed")

from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("./saved_bert_model")
MAX_LENGTH = 128
HF_ID_TO_LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def load_classifier_once():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing {MODEL_DIR} — run training first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    return tokenizer, model, device


st.set_page_config(page_title="AG News Topic Classifier", page_icon="📰", layout="centered")
st.title("📰 AG News Topic Classifier")
st.caption("Fine-tuned BERT (bert-base-uncased) — DevelopersHub Corporation")
st.markdown("Predicts: **World**, **Sports**, **Business**, or **Sci/Tech**.")
st.caption("Label map: 1 → World | 2 → Sports | 3 → Business | 4 → Sci/Tech")

if "clf" not in st.session_state:
    with st.spinner("Loading BERT model…"):
        st.session_state.clf = load_classifier_once()

tokenizer, model, device = st.session_state.clf

user_text = st.text_area(
    "News text",
    height=160,
    placeholder="e.g. Apple unveils new AI chip for next-generation MacBooks…",
)

if st.button("Classify", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text before classifying.")
    else:
        try:
            inputs = tokenizer(
                user_text.strip(),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item()) * 100.0
            topic = HF_ID_TO_LABEL.get(pred_id, "Unknown")

            st.success("Classification complete")
            c1, c2 = st.columns(2)
            c1.metric("Predicted Topic", topic)
            c2.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("All class probabilities")
            class_probs = {
                HF_ID_TO_LABEL[i]: float(probs[i].item()) * 100.0
                for i in range(len(probs))
            }
            for label_name, prob in sorted(class_probs.items(), key=lambda x: -x[1]):
                st.progress(min(prob / 100.0, 1.0), text=f"{label_name}: {prob:.2f}%")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
