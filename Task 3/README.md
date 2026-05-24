# Task 3 — Multimodal Housing Price Regression

**DevelopersHub Corporation — Internship Task 3**

Late-fusion **PyTorch** network predicting California housing prices from **tabular features** + **synthetic house images** ($64 \times 64$ RGB).

---

## Folder structure

```
Task 3/
├── README.md
└── multimodal_housing_price.ipynb    # End-to-end notebook (Cells 1–5)
```

No external image downloads required — images are generated on-the-fly in the `Dataset`.

---

## Architecture

```
Tabular [B, 8]  ──► MLP ──────────────────────► [B, 32]   ──┐
                                                             ├── cat ──► [B, 2080] ──► Head ──► [B]
Image [B,3,64,64] ──► CNN ──► flatten [4096] ──► Linear ──► [B, 2048] ┘
```

| Branch | Layers |
|--------|--------|
| **CNN** | 4× Conv2d→ReLU→MaxPool2d → Flatten (4096) → Linear → **2048-d** |
| **MLP** | Linear(8→64) → ReLU → BatchNorm → Linear(64→32) → ReLU → BatchNorm |
| **Fusion** | `torch.cat(dim=1)` → Linear(2080→256) → ReLU → Dropout → Linear(256→64) → ReLU → Linear(64→1) |

---

## Notebook sections

| Cell | Content |
|------|---------|
| **1** | Imports, California Housing, `StandardScaler` on X and y, synthetic image helper |
| **2** | `MultimodalHousingDataset`, 80/20 split, `DataLoader` (batch=32) |
| **3** | `MultimodalRegressionNet` with shape comments |
| **4** | 12 epochs, Adam 1e-3, MSE, train/val loss logging |
| **5** | MAE & RMSE (inverse-scaled), loss curve + predicted vs actual plot |

---

## Quick start

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib pillow jupyter
jupyter notebook multimodal_housing_price.ipynb
```

**Colab:** Runtime → GPU (optional). Run Cells 1–5 in order.

---

## Data

| Modality | Source |
|----------|--------|
| **Tabular** | `sklearn.datasets.fetch_california_housing` (20,640 samples, 8 features) |
| **Images** | Programmatic $64 \times 64$ RGB renders (walls/roof/windows tied to tabular fields) |

**Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

Target: median house value in **$100,000** units (e.g. `2.5` ≈ $250,000).

---

## Training defaults

| Setting | Value |
|---------|--------|
| Batch size | 32 |
| Epochs | 12 |
| Optimizer | Adam, lr = 1e-3 |
| Loss | MSELoss (on scaled target) |
| Train / val split | 80 / 20 (random) |

---

## Run results (verified pipeline execution)

### Cell 1 — Data preparation

| Metric | Value |
|--------|--------|
| Samples | 20,640 |
| Tabular features | 8 (scaled with `StandardScaler`) |
| Target shape | `(20640,)` scaled |
| Target (original) min / max / mean | **0.15** / **5.00** / **2.07** ($100k units) |

### Cell 2 — DataLoaders

| Split | Samples |
|-------|---------|
| Train | **16,512** |
| Validation | **4,128** |

| Tensor | Batch shape |
|--------|-------------|
| Tabular | `[32, 8]` |
| Image | `[32, 3, 64, 64]` |
| Target | `[32]` |

### Cell 3 — Model

- `MultimodalRegressionNet` instantiated successfully
- CNN flatten → **4096** features → Linear projection → **2048**
- MLP output → **32**
- Fusion input → **2080** → scalar price per sample
- Dummy forward: `torch.Size([2])`

### Cell 4 — Training (12 epochs)

| Epoch | Train MSE | Val MSE |
|-------|-----------|---------|
| 01 | 0.376224 | 0.293489 |
| 02 | 0.299519 | 0.264673 |
| 03 | 0.274934 | 0.255422 |
| 04 | 0.267818 | 0.240643 |
| 05 | 0.260758 | 0.295533 |
| 06 | 0.246652 | 0.252280 |
| 07 | 0.250306 | 0.216146 |
| 08 | 0.240678 | **0.212218** |
| 09 | 0.241851 | 0.237547 |
| 10 | 0.234253 | 0.324564 |
| 11 | 0.229675 | 0.229850 |
| 12 | 0.232951 | **0.211998** |

**Final epoch:** Train MSE **0.2330** | Val MSE **0.2120** (best validation at epoch 12)

Training loss decreased steadily; validation MSE fluctuated mid-training (epoch 10 spike) but ended at the run’s lowest value.

### Cell 5 — Validation metrics (original price scale)

| Metric | Value | Approx. USD |
|--------|--------|-------------|
| **MAE** | **0.3571** | ~**$35,700** |
| **RMSE** | **0.5313** | ~**$53,100** |

Cell 5 also displays:

1. **Loss curve** — training vs validation MSE over 12 epochs  
2. **Predicted vs actual scatter** — with ideal `y = x` reference line  

**Interpretation:** On held-out validation data, predictions are within ~$36k on average (MAE) in median house value terms. RMSE is higher due to larger errors on outlier properties—expected for this dataset. Multimodal fusion (tabular + synthetic images) learns a useful joint representation; synthetic images encode tabular-derived visual cues, so both branches contribute signal.

---

## Deliverables checklist

- [x] California Housing tabular load + scaling
- [x] Synthetic $64 \times 64$ image generation (no external upload)
- [x] Custom `Dataset` returning tabular, image, and target tensors
- [x] CNN + MLP late-fusion architecture with explicit shape flow
- [x] Training loop with epoch-wise train/val MSE
- [x] MAE & RMSE on inverse-scaled prices
- [x] Loss curve and predicted-vs-actual plots

---

## References

- [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [PyTorch nn.Module](https://pytorch.org/docs/stable/nn.html)
- [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html)
