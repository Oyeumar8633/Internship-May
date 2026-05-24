# Task 2 — Telco Customer Churn ML Pipeline

**DevelopersHub Corporation — Internship Task 2**

Modular, production-ready churn prediction using **scikit-learn `Pipeline`**, **`ColumnTransformer`**, and **`GridSearchCV`** on the IBM Telco Customer Churn dataset.

---

## Folder structure

```
Task 2/
├── README.md
├── telco_churn_pipeline.ipynb    # End-to-end notebook (Cells 1–6)
├── data/                         # Created at runtime (downloaded CSV)
│   └── Telco-Customer-Churn.csv
└── telco_churn_pipeline.joblib   # Created after Cell 6 (full fitted pipeline)
```

---

## Notebook sections

| Cell | Content |
|------|---------|
| **1** | Imports, logging, download & load Telco CSV |
| **2** | Clean `TotalCharges`, encode `Churn`, stratified 80/20 split |
| **3** | `ColumnTransformer`: numeric (median + scale), categorical (impute + one-hot) |
| **4** | Master pipeline + `GridSearchCV` (Logistic Regression vs Random Forest, `cv=5`, `f1`) |
| **5** | Test evaluation, classification report, confusion matrix & ROC plots |
| **6** | `joblib` save/load + raw-row inference verification |

---

## Quick start

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyter
jupyter notebook telco_churn_pipeline.ipynb
```

Run all cells in order. Grid search typically finishes in ~1–2 minutes on CPU (observed ~82s for 130 fits).

---

## Dataset source

[IBM Telco Customer Churn (icp4d)](https://github.com/IBM/telco-customer-churn-on-icp4d) — downloaded automatically in Cell 1 from:

`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

A legacy IBM mirror is kept as a fallback in the notebook if the primary URL fails.

---

## Run results (verified pipeline execution)

### Cell 1 — Data acquisition

| Metric | Value |
|--------|--------|
| Rows | 7,043 |
| Columns | 21 |
| Key fields | `customerID`, demographics, services, `MonthlyCharges`, `TotalCharges`, `Churn` |
| `TotalCharges` dtype (raw) | `object` (cleaned to numeric in Cell 2) |

### Cell 2 — Target & split

| Metric | Value |
|--------|--------|
| Churn rate (class 1) | **26.58%** |
| No churn (class 0) | **73.42%** |
| Train samples | 5,625 (80%) |
| Test samples | 1,407 (20%) |
| Train / test churn rate | 0.2658 / 0.2658 (stratified) |

### Cell 3 — Feature groups

**Numerical (4):** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`

**Categorical (15):** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

`customerID` is dropped before modeling. Preprocessing:

- **Numeric:** `SimpleImputer(median)` → `StandardScaler`
- **Categorical:** `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`

### Cell 4 — Grid search

| Setting | Value |
|---------|--------|
| CV folds | 5 |
| Scoring | `f1` |
| Candidates | 26 (Logistic Regression + Random Forest grids) |
| Total fits | 130 |
| Execution time | **~81.7 s** |
| Best CV F1 | **0.5965** |
| Best estimator | `LogisticRegression(max_iter=2000, solver='liblinear', random_state=42)` |

**Optimized hyperparameters:**

| Parameter | Value |
|-----------|--------|
| `classifier__C` | `1.0` |
| `classifier__penalty` | `l2` |

Random Forest was evaluated in the same `GridSearchCV` but logistic regression achieved the best cross-validated F1.

### Cell 5 — Test set evaluation

| Metric | Value |
|--------|--------|
| Overall accuracy | **0.80** |
| Weighted avg F1 | **0.80** |

**Classification report (test set):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No Churn | 0.85 | 0.89 | 0.87 | 1,033 |
| Churn | 0.65 | 0.57 | 0.61 | 374 |
| **Accuracy** | | | **0.80** | **1,407** |
| Macro avg | 0.75 | 0.73 | 0.74 | 1,407 |
| Weighted avg | 0.80 | 0.80 | 0.80 | 1,407 |

Cell 5 also displays an inline **confusion matrix** heatmap and **ROC curve** (see notebook outputs).

**Interpretation:** The pipeline generalizes well on the held-out test set. Performance is stronger on the majority class (no churn). Churn recall (0.57) is the main trade-off on this imbalanced dataset—typical without threshold tuning or `class_weight`.

### Cell 6 — Serialization & inference

| Item | Result |
|------|--------|
| Artifact | `telco_churn_pipeline.joblib` (preprocessors + best model) |
| Mock customer prediction | **Churn** (class `1`) |
| Churn probability | **0.6392** |

Raw feature dictionaries (no manual encoding) are passed through `pipeline.predict()` / `predict_proba()` after reload—confirming production-style inference.

---

## Production inference

After training, load the artifact and pass **raw feature dicts** (same columns as training, without `customerID` or `Churn`):

```python
import joblib
import pandas as pd

pipeline = joblib.load("telco_churn_pipeline.joblib")
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 1069.20,
}
row = pd.DataFrame([customer])
print("Class:", pipeline.predict(row))
print("Churn proba:", pipeline.predict_proba(row)[0, 1])
```

---

## Deliverables checklist

- [x] Automatic dataset download and load
- [x] EDA / cleaning (`TotalCharges`, binary `Churn`)
- [x] Stratified 80/20 train-test split
- [x] Modular `ColumnTransformer` preprocessing
- [x] `GridSearchCV` — Logistic Regression vs Random Forest
- [x] Test metrics, classification report, confusion matrix & ROC
- [x] `joblib` serialization + raw-row inference verification

---

## References

- [IBM Telco Customer Churn (icp4d)](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
