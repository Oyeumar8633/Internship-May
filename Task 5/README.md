# Task 5 — Automated Support Ticket Triage (LLM)

**DevelopersHub Corporation — Internship Task 5**

Zero-shot vs. few-shot **LLM triage** for free-text IT support tickets: predict and rank the **top 3** tags per ticket with structured JSON, evaluate against ground truth, and compare prompting strategies with charts and a summary report.

---

## Folder structure

```
Task 5/
├── README.md
├── ticketing_triage.ipynb           # End-to-end Colab notebook (Cells 1–5)
├── support_tickets_synthetic.csv    # 30-row dataset (Cell 1)
├── prompt_config.json               # Prompt templates + few-shot examples (Cell 2)
├── triage_evaluation_results.csv    # Per-ticket predictions (Cell 4)
├── triage_metrics_comparison.csv    # Zero-shot vs few-shot metrics (Cell 4)
└── figures/                         # Bar chart + confusion matrices (Cell 5)
    ├── accuracy_comparison_bar.png
    ├── confusion_zero_shot.png
    ├── confusion_few_shot.png
    └── misclassification_heatmap_few_shot.png
```

---

## Pipeline

```
Support ticket text
        │
        ▼
┌───────────────────┐     zero_shot or few_shot prompt
│  predict_tags()   │◄──── (JSON schema + optional 4 exemplars)
└─────────┬─────────┘
          │ LLM (Groq → Gemini → local Qwen)
          ▼
┌───────────────────┐
│ Parse JSON        │     top 3 tags + confidence + explanation
│ (regex salvage    │
│  if truncated)    │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Metrics           │     Top-1 / Top-3 vs ground_truth_tag
│ + visualizations  │
└───────────────────┘
```

---

## Target tags

| Tag | Example ticket themes |
|-----|------------------------|
| **Billing/Invoices** | Double charges, VAT, receipts, payment failures |
| **Account Access** | SSO, MFA, passwords, API keys, license expiry |
| **Technical Bug** | Crashes, HTTP 500, regressions, data/index issues |
| **Feature Request** | New integrations, exports, UI capabilities |
| **Hardware Issue** | Laptops, docks, UPS, printers, thermal throttling |

---

## Notebook cells

| Cell | Content |
|------|---------|
| **1** | Install packages; load `GROQ_API_KEY` / `GEMINI_API_KEY` from Colab Secrets; build **30** synthetic tickets → CSV |
| **2** | Zero-shot & few-shot prompt templates; **4** few-shot examples; save `prompt_config.json` |
| **3** | `LLMClient`, `predict_tags()`, robust JSON parsing; smoke test |
| **4** | Evaluate all tickets (zero-shot + few-shot); Top-1 / Top-3 accuracy; comparison table + CSVs |
| **5** | Bar chart, confusion matrices, misclassification heatmap, final summary report |

---

## Quick start (Google Colab)

1. Colab **Secrets** → `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)) — **recommended** (same key as Task 4).
2. Optional: `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) if not using Groq.
3. Upload `ticketing_triage.ipynb`.
4. **Runtime → Restart session** → **Run all** (Cells 1–5).

**LLM backend priority:** Groq (`llama-3.1-8b-instant`, JSON mode) → Gemini (`gemini-2.5-flash`) → local `Qwen/Qwen2.5-1.5B-Instruct`.

| Step | Approx. time |
|------|----------------|
| Cells 1–3 | ~1–2 min |
| Cell 4 | **~3–6 min** on Groq (60 API calls: 30 tickets × 2 modes) |
| Cell 5 | ~30 sec |

Without API keys, Cell 3 downloads Qwen locally — use a **GPU** runtime; CPU can take 30+ minutes for Cell 4.

---

## Dataset (Cell 1)

| Field | Description |
|-------|-------------|
| `ticket_id` | `TCK-1001` … `TCK-1030` |
| `created_at` | Synthetic timestamps |
| `channel` | email / portal / chat / phone |
| `priority` | low / medium / high / critical |
| `ticket_text` | Free-text support description |
| `ground_truth_tag` | Primary label for evaluation |

**Size:** 30 rows (6 per tag class). Saved as `support_tickets_synthetic.csv`.

---

## Prompting

| Mode | Description |
|------|-------------|
| **Zero-shot** | Tag list + strict JSON schema + ticket text only |
| **Few-shot** | Same as zero-shot plus **4** hand-written ticket → JSON exemplars |

Both modes require exactly **3** ranked tags from the fixed vocabulary with confidence scores and a short explanation.

---

## LLM JSON output schema

```json
{
  "top_tags": [
    {"rank": 1, "tag": "Technical Bug", "confidence": 0.85},
    {"rank": 2, "tag": "Account Access", "confidence": 0.10},
    {"rank": 3, "tag": "Feature Request", "confidence": 0.05}
  ],
  "explanation": "One or two sentences justifying the ranking."
}
```

`predict_tags()` tolerates markdown fences and **truncated** JSON (regex salvage on partial responses).

---

## Metrics

| Metric | Definition |
|--------|------------|
| **Top-1 accuracy** | Rank-1 predicted tag equals `ground_truth_tag` |
| **Top-3 accuracy** | Ground truth appears in any of the top 3 ranked tags |

---

## Run results

**Environment:** Google Colab, **Groq** `llama-3.1-8b-instant`, 30 synthetic tickets.

| Configuration | Top-1 accuracy | Top-3 accuracy | Correct (Top-1) |
|---------------|----------------|----------------|-----------------|
| **Zero-shot** | 76.7% | 100.0% | 23 / 30 |
| **Few-shot** | 90.0% | 100.0% | 27 / 30 |

**Lift (few-shot − zero-shot):**

| Metric | Lift |
|--------|------|
| Top-1 | **+13.3%** |
| Top-3 | **+0.0%** (already 100% in zero-shot) |

**Top-1 error count:** zero-shot **7**; few-shot **3**.

**Few-shot misclassification patterns (Top-1):**

| Ground truth | Predicted | Count |
|--------------|-----------|-------|
| Hardware Issue | Technical Bug | 2 |
| Billing/Invoices | Account Access | 1 |

### Interpretation

- **Few-shot** improved **Top-1** by giving the model concrete examples of tag boundaries and JSON format; ambiguous hardware vs. software wording benefited most.
- **Top-3** was already **100%** under zero-shot on this synthetic set, so few-shot did not raise Top-3 further — the true label was usually in the shortlist even when not ranked first.
- Remaining errors cluster on **overlapping language** (physical device vs. bug; billing vs. access), which is typical for real helpdesk triage.

### Cell 3 smoke test (example)

- **Ticket:** VPN SSO integration broke after weekend patch  
- **Top-1:** Technical Bug (confidence 0.8)  
- **Backend:** `groq | llama-3.1-8b-instant`

---

## Artifacts (after Run all)

| File | Description |
|------|-------------|
| `support_tickets_synthetic.csv` | Input dataset with labels |
| `prompt_config.json` | Serialized prompt templates and few-shot examples |
| `triage_evaluation_results.csv` | Per-ticket predictions, correctness flags, explanations |
| `triage_metrics_comparison.csv` | Aggregated zero-shot vs few-shot metrics |
| `figures/*.png` | Accuracy bar chart, confusion matrices, misclassification heatmap |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Gemini `429` / `limit: 0` | Use **`GROQ_API_KEY`** in Secrets; Groq is tried first |
| `No JSON object found` / truncated `{ "top_tags": [` | Re-run Cell 3 with latest notebook (salvage parser + higher Gemini token limit) or use Groq JSON mode |
| Cell 4 very slow | Confirm `LLM backend: groq` after Cell 3; enable GPU only if on local Qwen fallback |
| `VALID_TAGS` not defined | Run Cell 1 before Cell 2 |
| Cell 4 interrupted | Rerun Cell 4 only (Cells 1–3 must still be in kernel memory) |

---

## Dependencies (Cell 1)

`google-genai`, `groq`, `langchain-core`, `transformers`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

Install is quiet (`pip install -q` without upgrading Colab’s torch stack).
