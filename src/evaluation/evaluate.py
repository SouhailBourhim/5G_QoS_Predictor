"""
Evaluation Framework — src/evaluation/evaluate.py

8-pillar evaluation suite for the 5G QoS SLA violation classifiers.

Pillars
-------
1. verify_temporal_integrity   — assert no split overlap, violation rate ∈ (0.01, 0.15)
2. compute_classification_metrics — precision, recall, F1, AUC-ROC, AUC-PR, confusion matrix
3. compute_per_slice_metrics   — Pillar 2 per slice type
4. compute_per_event_recall    — recall per event_type label
5. compute_horizon_f1          — F1 at 15 / 30 / 60 min
6. compute_lead_time_stats     — median, mean, P25, P75 lead times for true positives
7. compute_shap_importance     — TreeExplainer; save bar plot to reports/figures/
8. compute_baseline_comparison — static 15%-of-threshold alerting baseline

run_evaluation orchestrates all pillars; Pillar 1 must pass before any other runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from src.models.classifier import load_classifier, _prepare_X_y, HORIZONS
from src.features.engineering import build_features
from src.data.splitter import temporal_split
from src.utils.config import SLICE_CONFIGS, MODELS_DIR


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

SLICE_TYPES = ["eMBB", "URLLC", "mMTC"]
HORIZON_MINS = [15, 30, 60]
VIOLATION_RATE_BOUNDS = (0.01, 0.15)
BASELINE_PROXIMITY_PCT = 0.15   # 15% proximity triggers static baseline alert
FIGURES_DIR = Path("reports/figures")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _load_test_splits(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Build feature matrices and return test splits for all slices."""
    file_map = {
        "eMBB":  raw_dir / "embb_synthetic.parquet",
        "URLLC": raw_dir / "urllc_synthetic.parquet",
        "mMTC":  raw_dir / "mmtc_synthetic.parquet",
    }
    slices_raw = {s: pd.read_parquet(p) for s, p in file_map.items() if p.exists()}
    test_splits = {}
    for stype, df_raw in slices_raw.items():
        others = {k: v for k, v in slices_raw.items() if k != stype}
        df_feat = build_features(df_raw, stype, others)
        _, _, test_df = temporal_split(df_feat)
        # Attach event_type from raw data aligned to test rows
        raw_test = df_raw.iloc[len(df_raw) - len(test_df):].reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if "event_type" in raw_test.columns:
            test_df["event_type"] = raw_test["event_type"].values
        test_splits[stype] = test_df
    return test_splits


# ─── PILLAR 1 ─────────────────────────────────────────────────────────────────

def verify_temporal_integrity(
    raw_dir: Path | str = "data/raw/generated",
) -> dict:
    """
    Pillar 1 — Assert:
      • No timestamp overlap between train/val/test splits
      • violation_in_30min rate ∈ (0.01, 0.15) for each split and slice

    Raises AssertionError on any failure (blocks all downstream pillars).
    Returns summary dict on success.
    """
    raw_dir = Path(raw_dir)
    file_map = {
        "eMBB":  raw_dir / "embb_synthetic.parquet",
        "URLLC": raw_dir / "urllc_synthetic.parquet",
        "mMTC":  raw_dir / "mmtc_synthetic.parquet",
    }
    results = {}
    for stype, path in file_map.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        train, val, test = temporal_split(df)

        # Overlap assertions
        assert train["timestamp"].max() < val["timestamp"].min(), \
            f"{stype}: train/val overlap!"
        assert val["timestamp"].max() < test["timestamp"].min(), \
            f"{stype}: val/test overlap!"

        # Violation rate assertions
        rates = {}
        for name, split in [("train", train), ("val", val), ("test", test)]:
            rate = split["violation_in_30min"].mean()
            lo, hi = VIOLATION_RATE_BOUNDS
            assert lo < rate < hi, \
                f"{stype}/{name}: violation_in_30min rate {rate:.3f} not in ({lo},{hi})"
            rates[name] = round(rate, 4)

        results[stype] = rates
    return results


# ─── PILLAR 2 ─────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Pillar 2 — Full binary classification metric set.

    Returns dict with: precision, recall, f1, auc_roc, auc_pr, confusion_matrix
    """
    return {
        "precision":        round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":               round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc":          round(float(roc_auc_score(y_true, y_proba)), 4),
        "auc_pr":           round(float(average_precision_score(y_true, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─── PILLAR 3 ─────────────────────────────────────────────────────────────────

def compute_per_slice_metrics(
    test_splits: dict[str, pd.DataFrame],
    horizon_min: int = 30,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """
    Pillar 3 — Classification metrics per slice type (at a single horizon).

    Returns {slice_type: metrics_dict}
    """
    results = {}
    for stype in SLICE_TYPES:
        if stype not in test_splits:
            continue
        clf, thr = load_classifier(stype, horizon_min, models_dir)
        X_test, y_test = _prepare_X_y(test_splits[stype], horizon_min)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= thr).astype(int)
        results[stype] = compute_classification_metrics(
            y_test.values, y_pred, y_proba
        )
    return results


# ─── PILLAR 4 ─────────────────────────────────────────────────────────────────

def compute_per_event_recall(
    test_splits: dict[str, pd.DataFrame],
    slice_type: str = "eMBB",
    horizon_min: int = 30,
    models_dir: Path = MODELS_DIR,
) -> dict[str, float]:
    """
    Pillar 4 — Recall per event_type label on the test split.

    Returns {event_type: recall_float}
    """
    df = test_splits[slice_type].copy()
    if "event_type" not in df.columns:
        return {}

    clf, thr = load_classifier(slice_type, horizon_min, models_dir)
    X_test, y_test = _prepare_X_y(df, horizon_min)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= thr).astype(int)
    df = df.iloc[:len(y_pred)].copy()
    df["_y_true"] = y_test.values
    df["_y_pred"] = y_pred

    results = {}
    for event in df["event_type"].unique():
        mask = df["event_type"] == event
        sub_true = df.loc[mask, "_y_true"]
        sub_pred = df.loc[mask, "_y_pred"]
        results[str(event)] = round(
            float(recall_score(sub_true, sub_pred, zero_division=0)), 4
        )
    return dict(sorted(results.items()))


# ─── PILLAR 5 ─────────────────────────────────────────────────────────────────

def compute_horizon_f1(
    test_splits: dict[str, pd.DataFrame],
    slice_type: str = "eMBB",
    models_dir: Path = MODELS_DIR,
) -> dict[int, float]:
    """
    Pillar 5 — F1 score at each prediction horizon (15/30/60 min).

    Returns {horizon_min: f1_score}
    """
    results = {}
    for h in HORIZON_MINS:
        clf, thr = load_classifier(slice_type, h, models_dir)
        X_test, y_test = _prepare_X_y(test_splits[slice_type], h)
        y_pred = (clf.predict_proba(X_test)[:, 1] >= thr).astype(int)
        results[h] = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    return results


# ─── PILLAR 6 ─────────────────────────────────────────────────────────────────

def compute_lead_time_stats(
    test_splits: dict[str, pd.DataFrame],
    slice_type: str = "eMBB",
    horizon_min: int = 30,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """
    Pillar 6 — Lead-time statistics for true positives.

    Lead time = time_to_violation at the moment the model fires an alert
    (true positive prediction). Only rows with a valid time_to_violation are used.

    Returns {median, mean, p25, p75} in minutes.
    """
    df = test_splits[slice_type].copy()
    clf, thr = load_classifier(slice_type, horizon_min, models_dir)
    X_test, y_test = _prepare_X_y(df, horizon_min)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= thr).astype(int)

    df = df.iloc[:len(y_pred)].reset_index(drop=True)
    df["_y_true"] = y_test.values
    df["_y_pred"] = y_pred

    # True positives with valid lead time
    tp_mask = (df["_y_true"] == 1) & (df["_y_pred"] == 1)
    if "time_to_violation" in df.columns:
        lead_times = df.loc[tp_mask, "time_to_violation"].dropna()
        lead_times = lead_times[lead_times > 0]
    else:
        lead_times = pd.Series(dtype=float)

    if len(lead_times) == 0:
        return {"median": None, "mean": None, "p25": None, "p75": None, "n_tp": 0}

    return {
        "median": round(float(lead_times.median()), 2),
        "mean":   round(float(lead_times.mean()),   2),
        "p25":    round(float(lead_times.quantile(0.25)), 2),
        "p75":    round(float(lead_times.quantile(0.75)), 2),
        "n_tp":   int(tp_mask.sum()),
    }


# ─── PILLAR 7 ─────────────────────────────────────────────────────────────────

def compute_shap_importance(
    test_splits: dict[str, pd.DataFrame],
    slice_type: str = "eMBB",
    horizon_min: int = 30,
    models_dir: Path = MODELS_DIR,
    figures_dir: Path = FIGURES_DIR,
    n_samples: int = 500,
) -> dict:
    """
    Pillar 7 — SHAP global feature importance via TreeExplainer.

    Uses a subsample of `n_samples` test rows for speed.
    Saves a bar chart to reports/figures/shap_{slice_type}_{horizon_min}min.png.

    Returns {feature: mean_abs_shap} for top-20.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clf, _ = load_classifier(slice_type, horizon_min, models_dir)
    X_test, _ = _prepare_X_y(test_splits[slice_type], horizon_min)

    # Sub-sample for speed
    X_sample = X_test.iloc[:n_samples]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_names = X_sample.columns.tolist()

    idx_top20 = np.argsort(mean_abs)[-20:][::-1]
    top_feats  = [feat_names[i] for i in idx_top20]
    top_scores = mean_abs[idx_top20]

    # Save bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_feats[::-1], top_scores[::-1], color="#4C9BE8")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Global Feature Importance — {slice_type} {horizon_min}min")
    plt.tight_layout()
    out_path = figures_dir / f"shap_{slice_type.lower()}_{horizon_min}min.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {top_feats[i]: round(float(top_scores[i]), 6) for i in range(len(top_feats))}


# ─── PILLAR 8 ─────────────────────────────────────────────────────────────────

def compute_baseline_comparison(
    test_splits: dict[str, pd.DataFrame],
    horizon_min: int = 30,
    models_dir: Path = MODELS_DIR,
    proximity_pct: float = BASELINE_PROXIMITY_PCT,
) -> dict:
    """
    Pillar 8 — Static threshold baseline comparison.

    Static baseline: alert when ANY KPI is within `proximity_pct`% of its SLA threshold.
    Compares precision, recall, F1 of XGBoost vs static baseline per slice.

    Returns {slice_type: {xgboost: metrics, baseline: metrics}}
    """
    results = {}

    for stype in SLICE_TYPES:
        if stype not in test_splits:
            continue

        df = test_splits[stype]
        target_col = HORIZONS[horizon_min]
        y_true = df[target_col].values

        # ── XGBoost metrics ────────────────────────────────────────────────
        clf, thr = load_classifier(stype, horizon_min, models_dir)
        X_test, y_test = _prepare_X_y(df, horizon_min)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_xgb   = (y_proba >= thr).astype(int)
        y_true_aligned = y_test.values

        xgb_metrics = {
            "precision": round(float(precision_score(y_true_aligned, y_xgb, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true_aligned, y_xgb, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true_aligned, y_xgb, zero_division=0)), 4),
        }

        # ── Static baseline ────────────────────────────────────────────────
        cfg = SLICE_CONFIGS.get(stype)
        baseline_alert = pd.Series(False, index=range(len(y_true)))

        if cfg is not None:
            for sla in cfg.sla_thresholds:
                # Map SLA kpi_name to DataFrame column name (strip unit suffix)
                col = sla.kpi_name.replace("_mbps", "").replace("_ms", "").replace("_pct", "")
                if col not in df.columns:
                    continue
                thresh = sla.threshold
                if sla.direction == "min" and thresh > 0:
                    # Alert when within proximity_pct of minimum (close to breaching)
                    proximity_val = thresh * (1 + proximity_pct)
                    baseline_alert |= pd.Series(df[col].values <= proximity_val)
                elif sla.direction == "max":
                    proximity_val = thresh * (1 - proximity_pct)
                    baseline_alert |= pd.Series(df[col].values >= proximity_val)

        y_base = baseline_alert.values.astype(int)

        baseline_metrics = {
            "precision": round(float(precision_score(y_true, y_base, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_base, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_base, zero_division=0)), 4),
        }

        results[stype] = {"xgboost": xgb_metrics, "baseline": baseline_metrics}

    return results


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def run_evaluation(
    raw_dir: str | Path = "data/raw/generated",
    models_dir: str | Path = MODELS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    shap_slice: str = "eMBB",
    shap_horizon: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Run all 8 evaluation pillars.

    Pillar 1 (temporal integrity) must pass before any other pillar executes.

    Returns a nested results dict with all pillar outputs.
    """
    raw_dir     = Path(raw_dir)
    models_dir  = Path(models_dir)
    figures_dir = Path(figures_dir)

    results = {}

    # ── Pillar 1 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 1: Verifying temporal integrity…")
    results["p1_temporal_integrity"] = verify_temporal_integrity(raw_dir)
    if verbose:
        print("  ✅ Passed")

    # Load test splits once for all downstream pillars
    if verbose:
        print("Loading test splits & building features…")
    test_splits = _load_test_splits(raw_dir)

    # ── Pillar 2 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 2: Computing classification metrics (eMBB 30min)…")
    clf, thr = load_classifier("eMBB", 30, models_dir)
    X_t, y_t = _prepare_X_y(test_splits["eMBB"], 30)
    y_p = clf.predict_proba(X_t)[:, 1]
    results["p2_classification"] = compute_classification_metrics(
        y_t.values, (y_p >= thr).astype(int), y_p
    )

    # ── Pillar 3 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 3: Per-slice metrics (30min horizon)…")
    results["p3_per_slice"] = compute_per_slice_metrics(test_splits, 30, models_dir)

    # ── Pillar 4 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 4: Per-event recall (eMBB 30min)…")
    results["p4_event_recall"] = compute_per_event_recall(
        test_splits, "eMBB", 30, models_dir
    )

    # ── Pillar 5 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 5: F1 per horizon (eMBB)…")
    results["p5_horizon_f1"] = {}
    for stype in SLICE_TYPES:
        results["p5_horizon_f1"][stype] = compute_horizon_f1(
            test_splits, stype, models_dir
        )

    # ── Pillar 6 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 6: Lead-time statistics…")
    results["p6_lead_time"] = {}
    for stype in SLICE_TYPES:
        results["p6_lead_time"][stype] = compute_lead_time_stats(
            test_splits, stype, 30, models_dir
        )

    # ── Pillar 7 ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"Pillar 7: SHAP importance ({shap_slice} {shap_horizon}min)…")
    results["p7_shap"] = compute_shap_importance(
        test_splits, shap_slice, shap_horizon, models_dir, figures_dir
    )

    # ── Pillar 8 ─────────────────────────────────────────────────────────────
    if verbose:
        print("Pillar 8: Baseline comparison (30min)…")
    results["p8_baseline"] = compute_baseline_comparison(
        test_splits, 30, models_dir
    )

    if verbose:
        print("\n✅ Evaluation complete.")

    return results


if __name__ == "__main__":
    import json
    res = run_evaluation(verbose=True)
    print("\n=== Summary ===")
    print(json.dumps(
        {k: v for k, v in res.items() if k != "p7_shap"}, indent=2
    ))
