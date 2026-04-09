"""
XGBoost SLA Violation Classifier — src/models/classifier.py

One model per (slice_type, horizon) — 9 models total (3 × 3).

Key design decisions
--------------------
- Uses `eval_metric='aucpr'` and `early_stopping_rounds=20`.
- `scale_pos_weight` = neg / pos from the training split to handle ~3–10% positive rate.
- `find_optimal_threshold` selects the *highest-precision* threshold on the validation
  set that still achieves recall ≥ 0.90.  Falls back to 0.5 if no threshold qualifies.
- Models saved as `{slice_type}_clf_{horizon}min.json` + `.threshold` text file.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

from src.data.splitter import temporal_split
from src.features.engineering import build_features
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

warnings.filterwarnings("ignore", category=UserWarning)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

HORIZONS = {15: "violation_in_15min", 30: "violation_in_30min", 60: "violation_in_60min"}

# Columns that must NEVER enter the feature matrix X
NON_FEATURE_COLS = {
    "timestamp", "slice_type", "event_type",
    "any_breach", "time_to_violation",
    "violation_in_15min", "violation_in_30min", "violation_in_60min",
}

XGB_BASE_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 20,
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric columns that are valid features (exclude targets etc.)."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _prepare_X_y(
    df: pd.DataFrame,
    horizon_min: int,
) -> tuple[pd.DataFrame, pd.Series]:
    target_col = HORIZONS[horizon_min]
    feat_cols = _get_feature_cols(df)
    X = df[feat_cols].astype("float32")
    y = df[target_col].astype(int)
    return X, y


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    slice_type: str,
    horizon_min: int,
    verbose: bool = True,
) -> XGBClassifier:
    """
    Train one XGBoost binary classifier.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val, y_val     : validation features and labels (used for early stopping)
    slice_type       : 'eMBB' | 'URLLC' | 'mMTC'
    horizon_min      : 15 | 30 | 60
    verbose          : print training summary

    Returns
    -------
    Trained XGBClassifier (already fitted).
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos if pos > 0 else 1.0

    if verbose:
        print(f"  [{slice_type} {horizon_min}min] "
              f"train pos={pos:,}/{len(y_train):,} ({pos/len(y_train)*100:.1f}%)  "
              f"scale_pos_weight={spw:.1f}")

    params = {**XGB_BASE_PARAMS, "scale_pos_weight": spw}
    clf = XGBClassifier(**params, verbosity=0)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return clf


def find_optimal_threshold(
    clf: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float = 0.90,
) -> float:
    """
    Select the highest-precision threshold on the validation set that still
    achieves recall ≥ `min_recall`.  Falls back to 0.5 if none qualifies.

    A threshold only qualifies if it:
      - achieves recall ≥ min_recall, AND
      - achieves precision > 0 (not a degenerate catch-all predictor)

    Parameters
    ----------
    clf       : fitted XGBClassifier
    X_val     : validation features
    y_val     : true validation labels
    min_recall: recall floor (default 0.90)

    Returns
    -------
    float threshold in [0, 1]
    """
    y_proba = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    # precision_recall_curve appends a sentinel at the end; align arrays
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # A valid threshold must achieve recall >= floor AND precision > 0
    mask = (recalls >= min_recall) & (precisions > 0)
    if not mask.any():
        return 0.5

    # Among qualifying thresholds, pick highest precision
    best_idx = int(np.argmax(precisions[mask]))
    qualifying_thresholds = thresholds[mask]
    return float(qualifying_thresholds[best_idx])


def save_classifier(
    clf: XGBClassifier,
    threshold: float,
    slice_type: str,
    horizon_min: int,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Save model JSON and threshold float to `models_dir`."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slice_type.lower()}_clf_{horizon_min}min"
    clf.save_model(models_dir / f"{stem}.json")
    (models_dir / f"{stem}.threshold").write_text(str(threshold))


def load_classifier(
    slice_type: str,
    horizon_min: int,
    models_dir: Path = MODELS_DIR,
) -> tuple[XGBClassifier, float]:
    """Load model + threshold.  Returns (XGBClassifier, float)."""
    models_dir = Path(models_dir)
    stem = f"{slice_type.lower()}_clf_{horizon_min}min"
    clf = XGBClassifier()
    clf.load_model(models_dir / f"{stem}.json")
    threshold = float((models_dir / f"{stem}.threshold").read_text())
    return clf, threshold


# ─── BATCH TRAINING ───────────────────────────────────────────────────────────

def train_all_classifiers(
    raw_dir: Path | str = "data/raw/generated",
    models_dir: Path | str = MODELS_DIR,
    verbose: bool = True,
) -> dict:
    """
    End-to-end: load raw Parquets → feature engineering → split → train 9 models.

    Returns a nested dict: {slice_type: {horizon_min: (clf, threshold)}}
    """
    raw_dir = Path(raw_dir)
    models_dir = Path(models_dir)
    results = {}

    file_map = {
        "eMBB": raw_dir / "embb_synthetic.parquet",
        "URLLC": raw_dir / "urllc_synthetic.parquet",
        "mMTC": raw_dir / "mmtc_synthetic.parquet",
    }

    # Load all slices for cross-slice features
    slices_raw: dict[str, pd.DataFrame] = {}
    for stype, path in file_map.items():
        if path.exists():
            slices_raw[stype] = pd.read_parquet(path)

    for stype, df_raw in slices_raw.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Slice: {stype}")
            print(f"{'='*60}")

        others = {k: v for k, v in slices_raw.items() if k != stype}
        df_feat = build_features(df_raw, stype, others)

        train_df, val_df, test_df = temporal_split(df_feat)

        results[stype] = {}
        for horizon_min in [15, 30, 60]:
            X_train, y_train = _prepare_X_y(train_df, horizon_min)
            X_val,   y_val   = _prepare_X_y(val_df,   horizon_min)

            clf = train_classifier(X_train, y_train, X_val, y_val,
                                   stype, horizon_min, verbose=verbose)
            threshold = find_optimal_threshold(clf, X_val, y_val)

            if verbose:
                print(f"  [{stype} {horizon_min}min] "
                      f"best threshold={threshold:.3f}  "
                      f"best_iteration={clf.best_iteration}")

            save_classifier(clf, threshold, stype, horizon_min, models_dir)
            results[stype][horizon_min] = (clf, threshold)

    return results


if __name__ == "__main__":
    train_all_classifiers(verbose=True)
