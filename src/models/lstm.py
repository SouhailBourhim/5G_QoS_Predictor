"""
Optional LSTM SLA Violation Classifier — src/models/lstm.py

Architecture (per design spec §7):
  Input: (batch, seq_len=24, n_features)
    → 2-layer LSTM (hidden=128, dropout=0.2)
    → last hidden state: (batch, 128)
    → Linear(128→64) → ReLU
    → Linear(64→32)  → ReLU
    → Linear(32→1)   → Sigmoid
  Output: (batch,) violation probability

Key constraints
---------------
- Sliding window stride=1, seq_len=24 (2 hours of 5-min data).
- No random shuffling of the sequence — temporal order is always preserved.
- Uses the same temporal splits (train/val/test) as the XGBoost classifier.
- Checkpoints saved to models/lstm_{slice_type}_{horizon}min.pt

Usage
-----
    from src.models.lstm import train_lstm
    train_lstm("eMBB", horizon_min=30)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.splitter import temporal_split
from src.features.engineering import build_features
from src.models.classifier import _get_feature_cols, HORIZONS
from src.utils.config import MODELS_DIR


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

SEQ_LEN = 24          # 2 hours of 5-minute intervals
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5          # early stopping patience on val AUC-PR


# ─── DATASET ─────────────────────────────────────────────────────────────────

class SLASequenceDataset(Dataset):
    """
    Sliding window dataset for LSTM training.

    For a DataFrame with N rows, produces N - seq_len samples:
      X[i] = features[i : i+seq_len]       shape: (seq_len, n_features)
      y[i] = target[i + seq_len - 1]       scalar label at the last timestep

    Temporal order is preserved; no shuffling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        horizon_min: int,
        seq_len: int = SEQ_LEN,
    ):
        target_col = HORIZONS[horizon_min]
        feat_cols  = _get_feature_cols(df)

        X_raw = df[feat_cols].astype("float32")
        y_raw = df[target_col].astype("float32")

        # Replace inf with NaN then fill with column mean; drop remaining NaNs
        X_raw = X_raw.replace([float("inf"), float("-inf")], float("nan"))
        col_means = X_raw.mean()
        X_raw = X_raw.fillna(col_means)

        # Align rows
        valid = X_raw.notna().all(axis=1) & y_raw.notna()
        X_np = X_raw[valid].values   # (N, F)
        y_np = y_raw[valid].values   # (N,)

        self.X = torch.from_numpy(X_np)
        self.y = torch.from_numpy(y_np)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx : idx + self.seq_len]          # (seq_len, F)
        y = self.y[idx + self.seq_len - 1]            # scalar
        return x, y


# ─── MODEL ────────────────────────────────────────────────────────────────────

class SLAViolationLSTM(nn.Module):
    """
    2-layer LSTM + FC head for binary SLA violation classification.

    Parameters
    ----------
    input_size  : number of input features (set at runtime from the feature matrix)
    hidden_size : LSTM hidden dimension (default 128)
    num_layers  : number of stacked LSTM layers (default 2)
    dropout     : dropout between LSTM layers (default 0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # No Sigmoid here — raw logits; BCEWithLogitsLoss handles it
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (hidden, _) = self.lstm(x)      # hidden: (num_layers, batch, hidden_size)
        last_hidden = hidden[-1]            # (batch, hidden_size)
        return self.fc(last_hidden).squeeze(-1)  # (batch,) — raw logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities in [0,1]."""
        return torch.sigmoid(self.forward(x))


# ─── TRAINING UTILITIES ───────────────────────────────────────────────────────

def _pos_weight(y_series: pd.Series) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight tensor."""
    neg = int((y_series == 0).sum())
    pos = int((y_series == 1).sum())
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


def save_lstm(model: SLAViolationLSTM, slice_type: str, horizon_min: int,
              models_dir: Path = MODELS_DIR) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"lstm_{slice_type.lower()}_{horizon_min}min.pt"
    torch.save(model.state_dict(), path)


def load_lstm(input_size: int, slice_type: str, horizon_min: int,
              models_dir: Path = MODELS_DIR) -> SLAViolationLSTM:
    path = Path(models_dir) / f"lstm_{slice_type.lower()}_{horizon_min}min.pt"
    model = SLAViolationLSTM(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ─── END-TO-END TRAINING ─────────────────────────────────────────────────────

def train_lstm(
    slice_type: str,
    horizon_min: int,
    raw_dir: str | Path = "data/raw/generated",
    models_dir: str | Path = MODELS_DIR,
    seq_len: int = SEQ_LEN,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> SLAViolationLSTM:
    """
    Full pipeline: load raw data → features → split → train LSTM.

    Returns the trained model (also saved to models_dir).
    """
    raw_dir = Path(raw_dir)
    file_names = {
        "eMBB":  "embb_synthetic.parquet",
        "URLLC": "urllc_synthetic.parquet",
        "mMTC":  "mmtc_synthetic.parquet",
    }

    # ── Build feature matrices ────────────────────────────────────────────────
    slices_raw = {s: pd.read_parquet(raw_dir / f) for s, f in file_names.items()
                  if (raw_dir / f).exists()}
    df_raw    = slices_raw[slice_type]
    others    = {k: v for k, v in slices_raw.items() if k != slice_type}
    df_feat   = build_features(df_raw, slice_type, others)

    train_df, val_df, _ = temporal_split(df_feat)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = SLASequenceDataset(train_df, horizon_min, seq_len)
    val_ds   = SLASequenceDataset(val_df,   horizon_min, seq_len)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    n_features = train_ds.X.shape[1]

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[LSTM] {slice_type} {horizon_min}min  "
              f"n_features={n_features}  device={device}")

    model     = SLAViolationLSTM(input_size=n_features).to(device)
    pos_w     = _pos_weight(train_df[HORIZONS[horizon_min]]).to(device)
    # BCEWithLogitsLoss: numerically stable, applies sigmoid internally
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, max_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)          # (batch,) probabilities in [0,1]
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if verbose:
            print(f"  Epoch {epoch:02d}/{max_epochs}  "
                  f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val_loss={best_val_loss:.4f})")
                break

    # Restore best weights and save
    if best_state:
        model.load_state_dict(best_state)
    model.to("cpu")
    save_lstm(model, slice_type, horizon_min, models_dir)
    if verbose:
        print(f"  Saved to models/lstm_{slice_type.lower()}_{horizon_min}min.pt")
    return model


def train_all_lstm(
    slice_types: list[str] | None = None,
    horizons: list[int] | None = None,
    **kwargs,
) -> dict:
    """Train LSTM classifiers for the specified slices and horizons."""
    if slice_types is None:
        slice_types = ["eMBB", "URLLC", "mMTC"]
    if horizons is None:
        horizons = [15, 30, 60]

    results = {}
    for stype in slice_types:
        results[stype] = {}
        for h in horizons:
            model = train_lstm(stype, h, **kwargs)
            results[stype][h] = model
    return results


if __name__ == "__main__":
    # Train eMBB 30min as the primary demonstration model
    train_lstm("eMBB", horizon_min=30, verbose=True)
