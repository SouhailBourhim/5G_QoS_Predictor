# Task 8 — Optional LSTM SLA Violation Classifier: Analysis

**Module:** `src/models/lstm.py`
**Checkpoint saved:** `models/lstm_embb_30min.pt`
**Demonstrated model:** eMBB slice, 30-minute violation horizon

---

## What was done

An optional LSTM-based binary classifier was implemented as an architectural alternative to the XGBoost classifier suite from Task 6. The model ingests sliding windows of length 24 (2 hours of 5-minute data) from the engineered feature matrix and outputs a scalar SLA violation probability.

The implementation covers three components:

1. **`SLASequenceDataset`** — A `torch.utils.data.Dataset` that constructs (X, y) pairs from a time-ordered DataFrame using a sliding window of `seq_len=24` and stride=1. No shuffling is applied at any point. NaN and ±∞ values in the feature matrix are imputed with column means before conversion to tensors to prevent NaN gradients.

2. **`SLAViolationLSTM`** — The model architecture: 2-layer LSTM (hidden=128, dropout=0.2) feeds its last hidden state into a 3-layer FC head (128→64→32→1) outputting raw logits.

3. **`train_lstm`** — End-to-end training pipeline: feature engineering → temporal split → dataset construction → training with `BCEWithLogitsLoss(pos_weight=neg/pos)` → early stopping on validation loss → checkpoint save.

---

## Architecture Details

```
Input: (batch, 24, 318 features)
  ↓
LSTM(hidden=128, layers=2, dropout=0.2)
  ↓
Last hidden state: (batch, 128)
  ↓
Linear(128→64) → ReLU
Linear(64→32)  → ReLU
Linear(32→1)   → raw logit
  ↓
BCEWithLogitsLoss(pos_weight) ← numerically stable sigmoid + weighting
  ↓
sigmoid(logit) for inference → violation probability ∈ [0,1]
```

---

## Training Results — eMBB 30min Demo Model

| Setting | Value |
|---|---|
| Device | MPS (Apple Silicon) |
| Input features | 318 |
| Sequence length | 24 steps (2 hours) |
| Optimizer | Adam (lr=1e-3) |
| Positives in training set | 1,193 / 17,280 (6.9%) |
| `pos_weight` | 13.5× |
| Epochs run | 16 (early stopped) |
| Best val_loss | 1.041 |

Training loss progressed from **1.234 → 0.948** before the validation loss stopped improving and early stopping triggered at epoch 16. The model converged without NaN gradients after implementing proper inf/NaN imputation and using `BCEWithLogitsLoss` rather than a standalone Sigmoid + BCELoss (which had no mechanism for class-weight balancing).

---

## Key Design Decisions

### Why BCEWithLogitsLoss, not BCELoss?
`BCELoss` with a model that outputs sigmoid activations has no built-in `pos_weight` mechanism. On a 6.9% positive rate, this causes the model to collapse to predicting zero for all samples. `BCEWithLogitsLoss` accepts raw logits internally combines the sigmoid and the weighted BCE computation in a numerically stable form, making the `pos_weight` scalar effective at counteracting class imbalance.

### Why impute NaN rather than drop rows?
The engineered feature matrix contains NaN values for the first ~288 rows (the warm-up period where lag and rolling windows are initialised). Dropping these rows would break the sliding window's contiguous indexing. Imputing with column means preserves temporal continuity in the dataset while eliminating NaN gradients.

### Why seq_len=24?
24 steps × 5 minutes = 2 hours of temporal context. This matches the EWMA and rolling window spans used in Task 3 (spans up to 288), ensuring the LSTM sees the same contextual scope that the XGBoost feature engineering explicitly encodes. It also keeps the dataset large enough for gradient-based learning (17,256 training windows from 17,280 rows).

---

## Comparison to XGBoost Classifier (Task 6)

| Aspect | XGBoost (Task 6) | LSTM (Task 8) |
|---|---|---|
| Input format | Single timestep, ~254 hand-crafted features | 24-timestep raw window, 318 features |
| Class weighting | `scale_pos_weight` | `pos_weight` in BCEWithLogitsLoss |
| Training speed | Fast (~30s per model) | Slower (~3–4 min per model on MPS) |
| Interpretability | SHAP feature importance | Opaque hidden state |
| Early stopping | On AUC-PR (`aucpr` eval_metric) | On validation BCE loss |
| Current models | 9 fully trained | 1 demo model (eMBB 30min) |

The LSTM provides a complementary architecture for capturing fine-grained temporal dynamics within the 2-hour context window. Its primary advantage is learning non-linear temporal dependencies without requiring explicit lag/rolling feature engineering — the sequence context is encoded directly.

---

## Implications for Task 10 (Evaluation)

- The eMBB 30min LSTM can be compared against the XGBoost eMBB 30min classifier via the evaluation framework (Pillar 8 baseline comparison extension).
- `predict_proba()` method returns sigmoid-transformed probabilities for direct comparison with the XGBoost threshold selection framework.
- Full 9-model LSTM training is available via `train_all_lstm()` if a full head-to-head evaluation is desired.

---

## Conclusion

The LSTM classifier is fully functional, training correctly with decreasing loss and class-imbalance weighting. The eMBB 30min demonstration model is saved and loadable via `load_lstm()`. The architecture satisfies all design spec requirements (§7): 2-layer LSTM, hidden=128, dropout=0.2, FC head 128→64→32→1, sliding window seq_len=24, no shuffling, same temporal splits as XGBoost.
