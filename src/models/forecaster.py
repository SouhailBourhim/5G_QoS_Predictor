"""
XGBoost KPI Forecaster — src/models/forecaster.py

One regression model per (slice_type, kpi, horizon) — up to 63 models.
Target: future KPI value at time t+h, constructed as df[kpi].shift(-h).

Horizons map to step counts (5-minute intervals):
  15 min → 3 steps
  30 min → 6 steps
  60 min → 12 steps

Usage
-----
    from src.models.forecaster import train_all_forecasters
    train_all_forecasters()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data.splitter import temporal_split
from src.features.engineering import build_features
from src.models.classifier import _get_feature_cols, NON_FEATURE_COLS
from src.utils.config import MODELS_DIR


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# horizon_minutes → forecast steps (each 5 min)
HORIZON_STEPS = {15: 3, 30: 6, 60: 12}

# KPIs present in the raw synthetic data
KPI_COLS = [
    "dl_throughput", "latency", "jitter",
    "packet_loss", "prb_util", "active_users", "reliability",
]

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
    "eval_metric": "rmse",
    "early_stopping_rounds": 20,
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _make_regression_target(df: pd.DataFrame, kpi: str, horizon_steps: int) -> pd.Series:
    """Shift KPI column forward by `horizon_steps` to create the future-value target.
    Rows at the tail where the future value is unavailable are set to NaN."""
    return df[kpi].shift(-horizon_steps)


def _prepare_fcst_X_y(
    df: pd.DataFrame,
    kpi: str,
    horizon_steps: int,
) -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = _get_feature_cols(df)
    target = _make_regression_target(df, kpi, horizon_steps)
    # Drop rows where the future target is undefined (last `horizon_steps` rows)
    mask = target.notna()
    X = df[feat_cols].loc[mask].astype("float32")
    y = target.loc[mask].astype("float32")
    return X, y


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def train_forecaster(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    slice_type: str,
    kpi: str,
    horizon_steps: int,
    verbose: bool = False,
) -> XGBRegressor:
    """
    Train one XGBoost regression model to forecast a KPI `horizon_steps` ahead.

    Parameters
    ----------
    X_train, y_train : training features and future-value targets
    X_val, y_val     : validation features and targets (used for early stopping)
    slice_type       : 'eMBB' | 'URLLC' | 'mMTC'
    kpi              : name of the KPI being forecast
    horizon_steps    : number of 5-minute steps ahead (3, 6, or 12)

    Returns
    -------
    Fitted XGBRegressor.
    """
    model = XGBRegressor(**XGB_BASE_PARAMS, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_forecaster(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a trained forecaster on the test split.

    Returns
    -------
    dict with keys: mae, rmse, mape
    """
    y_pred = model.predict(X_test)
    y_true = y_test.values

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # MAPE: skip rows where true value is ~0 to avoid division by zero
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2)}


def save_forecaster(
    model: XGBRegressor,
    slice_type: str,
    kpi: str,
    horizon_min: int,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Save model to `models/{slice_type}_fcst_{kpi}_{horizon_min}min.json`."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slice_type.lower()}_fcst_{kpi}_{horizon_min}min"
    model.save_model(models_dir / f"{stem}.json")


def load_forecaster(
    slice_type: str,
    kpi: str,
    horizon_min: int,
    models_dir: Path = MODELS_DIR,
) -> XGBRegressor:
    """Load a saved forecaster model."""
    stem = f"{slice_type.lower()}_fcst_{kpi}_{horizon_min}min"
    model = XGBRegressor()
    model.load_model(Path(models_dir) / f"{stem}.json")
    return model


# ─── BATCH TRAINING ───────────────────────────────────────────────────────────

def train_all_forecasters(
    raw_dir: Path | str = "data/raw/generated",
    models_dir: Path | str = MODELS_DIR,
    verbose: bool = True,
) -> dict:
    """
    End-to-end: load raw Parquets → feature engineering → split → train models.

    Skips KPIs that don't exist in a given slice's DataFrame.

    Returns a nested dict: {slice_type: {kpi: {horizon_min: {'model', 'metrics'}}}}
    """
    raw_dir = Path(raw_dir)
    models_dir = Path(models_dir)
    results = {}

    file_map = {
        "eMBB":  raw_dir / "embb_synthetic.parquet",
        "URLLC": raw_dir / "urllc_synthetic.parquet",
        "mMTC":  raw_dir / "mmtc_synthetic.parquet",
    }

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
        kpis_in_slice = [k for k in KPI_COLS if k in df_feat.columns]

        for kpi in kpis_in_slice:
            results[stype][kpi] = {}
            for horizon_min, horizon_steps in HORIZON_STEPS.items():
                X_train, y_train = _prepare_fcst_X_y(train_df, kpi, horizon_steps)
                X_val,   y_val   = _prepare_fcst_X_y(val_df,   kpi, horizon_steps)
                X_test,  y_test  = _prepare_fcst_X_y(test_df,  kpi, horizon_steps)

                model = train_forecaster(
                    X_train, y_train, X_val, y_val,
                    stype, kpi, horizon_steps, verbose=False,
                )
                metrics = evaluate_forecaster(model, X_test, y_test)
                save_forecaster(model, stype, kpi, horizon_min, models_dir)

                results[stype][kpi][horizon_min] = {"metrics": metrics}

                if verbose:
                    print(
                        f"  [{stype}] {kpi:<18} {horizon_min}min  "
                        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
                        f"MAPE={metrics['mape']:.2f}%  "
                        f"iter={model.best_iteration}"
                    )

    return results


if __name__ == "__main__":
    train_all_forecasters(verbose=True)
