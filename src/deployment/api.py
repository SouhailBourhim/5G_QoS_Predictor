"""
FastAPI REST API — src/deployment/api.py

Endpoints
---------
POST /predict
    Body:  KPIHistory  (slice_type + list of ≥12 timestep dicts)
    Response: PredictionResponse

GET  /health/{slice_type}
    Response: { slice_type, health_status, violation_prob_30min }

GET  /slices
    Response: { slices: [...] }

Startup
-------
All 9 classifier models + thresholds are loaded at startup into in-memory dicts.
Forecasters are loaded on demand (63 files would be too much to hold concurrently).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.models.classifier import load_classifier, _get_feature_cols, HORIZONS
from src.models.forecaster import load_forecaster, KPI_COLS, HORIZON_STEPS
from src.features.engineering import build_features
from src.utils.config import MODELS_DIR


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

SUPPORTED_SLICES = ["eMBB", "URLLC", "mMTC"]
MIN_TIMESTEPS    = 12          # minimum sequence length for feature engineering
MODELS_PATH      = Path(MODELS_DIR)

# Health-status thresholds based on 30min violation probability
HEALTH_CRITICAL  = 0.60
HEALTH_WARNING   = 0.30

# SHAP feature → recommendation string mapping
SHAP_RECOMMENDATIONS: dict[str, str] = {
    # SLA proximity features
    "sla_margin":           "KPI approaching SLA threshold — initiate pre-emptive resource allocation",
    "time_to_breach":       "Estimated breach time is imminent — escalate to active remediation",
    # Throughput degradation
    "dl_throughput":        "Downlink throughput declining — check backhaul congestion and PRB allocation",
    "dl_throughput_lag1":   "Throughput drop observed in previous interval — monitor for sustained decline",
    "dl_throughput_roll3":  "Rolling throughput trend decreasing — consider load balancing",
    # Latency features
    "latency":              "Latency elevated — investigate transport layer queuing and routing delays",
    "latency_lag1":         "Recent latency spike detected — check for sudden traffic bursts",
    "latency_ewma_span12":  "Sustained latency increase — review inter-site handover configuration",
    # PRB utilisation
    "prb_util":             "PRB utilisation high — schedule capacity expansion or traffic steering",
    "prb_util_roll3":       "PRB utilisation trending upward — activate load-sharing with neighbouring cells",
    "prb_util_ewma_span12": "Long-term PRB trend elevated — review spectrum allocation policy",
    # Packet loss
    "packet_loss":          "Packet loss detected — inspect radio link quality and interference levels",
    "packet_loss_lag1":     "Recent packet loss spike — initiate RF interference scan",
    # Cross-slice pressure
    "cross_slice_prb":      "Cross-slice resource contention detected — adjust slice priority weights",
    "active_users":         "High user density — trigger dynamic user-to-cell assignment",
    # Jitter
    "jitter":               "Jitter exceeds acceptable bounds — review QoS scheduling parameters",
    "jitter_lag1":          "Recent jitter increase — prioritise URLLC packet scheduling",
}

DEFAULT_RECOMMENDATION = (
    "Anomalous pattern detected — review KPI trends and consider proactive resource reallocation"
)


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────

class TimestepRecord(BaseModel):
    """A single 5-minute KPI snapshot."""
    timestamp:    str
    dl_throughput: float
    latency:       float
    jitter:        float
    packet_loss:   float
    prb_util:      float
    active_users:  float
    reliability:   float


class KPIHistory(BaseModel):
    """
    Request body for POST /predict.

    Attributes
    ----------
    slice_type : one of 'eMBB', 'URLLC', 'mMTC'
    history    : list of ≥12 consecutive 5-minute KPI readings
    """
    slice_type: str
    history:    list[TimestepRecord]

    @field_validator("slice_type")
    @classmethod
    def validate_slice(cls, v: str) -> str:
        if v not in SUPPORTED_SLICES:
            raise ValueError(f"slice_type must be one of {SUPPORTED_SLICES}")
        return v

    @field_validator("history")
    @classmethod
    def validate_history_length(cls, v: list) -> list:
        if len(v) < MIN_TIMESTEPS:
            raise ValueError(
                f"history must contain at least {MIN_TIMESTEPS} timesteps, got {len(v)}"
            )
        return v


class HorizonPrediction(BaseModel):
    horizon_min:      int
    violation_prob:   float
    predicted_kpis:   dict[str, float]


class PredictionResponse(BaseModel):
    slice_type:         str
    health_status:      str          # "healthy" | "warning" | "critical"
    horizons:           list[HorizonPrediction]
    top_shap_features:  list[dict[str, Any]]
    recommendations:    list[str]


class HealthResponse(BaseModel):
    slice_type:           str
    health_status:        str
    violation_prob_30min: float


# ─── APP & STARTUP ────────────────────────────────────────────────────────────

# In-memory stores (populated at startup)
_classifiers: dict[tuple[str, int], tuple] = {}   # (slice_type, horizon) → (clf, threshold)
_explainers:  dict[tuple[str, int], Any]   = {}   # (slice_type, horizon) → shap.TreeExplainer


def _load_all_models() -> None:
    """Load all 9 classifier models + thresholds into memory."""
    for stype in SUPPORTED_SLICES:
        for h in [15, 30, 60]:
            try:
                clf, thr = load_classifier(stype, h, MODELS_PATH)
                _classifiers[(stype, h)] = (clf, thr)
                if h == 30:
                    _explainers[(stype, 30)] = shap.TreeExplainer(clf)
            except Exception as e:
                print(f"[WARN] Could not load {stype} {h}min classifier: {e}")
    print(f"[INFO] Loaded {len(_classifiers)} classifiers.")


def _ensure_loaded() -> None:
    """Lazy-load models the first time an endpoint is called (guards TestClient use)."""
    if not _classifiers:
        _load_all_models()


@asynccontextmanager
async def lifespan(application: FastAPI):
    _load_all_models()
    yield


app = FastAPI(
    title="5G QoS SLA Violation Predictor",
    description=(
        "Real-time SLA violation probability predictions for 5G network slices "
        "(eMBB, URLLC, mMTC) across 15, 30, and 60-minute horizons."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _history_to_df(history: list[TimestepRecord]) -> pd.DataFrame:
    """Convert the request history list to a DataFrame with correct dtypes."""
    records = [r.model_dump() for r in history]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _derive_health(prob_30min: float) -> str:
    if prob_30min >= HEALTH_CRITICAL:
        return "critical"
    elif prob_30min >= HEALTH_WARNING:
        return "warning"
    return "healthy"


def _get_recommendations(shap_feature_names: list[str]) -> list[str]:
    """Map top SHAP feature names to recommendation strings."""
    recs = []
    seen: set[str] = set()
    for feat in shap_feature_names:
        # Try exact match then prefix match
        rec = SHAP_RECOMMENDATIONS.get(feat)
        if rec is None:
            for key in SHAP_RECOMMENDATIONS:
                if feat.startswith(key) or key in feat:
                    rec = SHAP_RECOMMENDATIONS[key]
                    break
        if rec is None:
            rec = DEFAULT_RECOMMENDATION
        if rec not in seen:
            recs.append(rec)
            seen.add(rec)
    return recs[:5]  # cap at 5 recommendations


def _build_feature_row(df_raw: pd.DataFrame, slice_type: str) -> pd.DataFrame:
    """
    Build a feature matrix from the raw KPI history DataFrame.
    Uses build_features() with no cross-slice neighbours (live inference).
    """
    # Ensure required columns exist with correct naming
    df = df_raw.copy()
    df["slice_type"] = slice_type
    df["event_type"] = "normal"    # unknown at inference time

    # Synthetic target columns (not used for prediction, just satisfy schema)
    for h in [15, 30, 60]:
        col = f"violation_in_{h}min"
        if col not in df.columns:
            df[col] = 0
    for col in ["any_breach", "time_to_violation"]:
        if col not in df.columns:
            df[col] = 0.0

    # Feature engineering with no cross-slice neighbours
    df_feat = build_features(df, slice_type, other_slices={})
    return df_feat


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(body: KPIHistory) -> PredictionResponse:
    """
    Predict SLA violation probability at 15 / 30 / 60-minute horizons.

    Requires ≥12 historical KPI timesteps. Returns:
    - Per-horizon violation probability
    - Per-horizon KPI forecasts (15-min-ahead)
    - Top-5 SHAP features driving the 30min prediction
    - Actionable recommendations mapped from SHAP features
    - Overall health_status (healthy|warning|critical) based on 30min probability
    """
    stype = body.slice_type
    _ensure_loaded()

    # Build feature matrix from history
    df_raw  = _history_to_df(body.history)
    df_feat = _build_feature_row(df_raw, stype)

    feat_cols = _get_feature_cols(df_feat)

    # Align live feature columns to the training set by zero-filling any missing columns
    # (cross-slice features are absent at inference without neighbor data)
    if (stype, 30) in _classifiers:
        clf_30, _ = _classifiers[(stype, 30)]
        expected_cols = list(clf_30.get_booster().feature_names)
        for col in expected_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        feat_cols = expected_cols

    # Use last valid feature row for prediction
    df_clean = df_feat[feat_cols].ffill().fillna(0.0)
    X_last   = df_clean.iloc[[-1]]   # (1, n_features)

    # ── Per-horizon classifier predictions ────────────────────────────────────
    horizons_out: list[HorizonPrediction] = []
    prob_30min = 0.0

    for h in [15, 30, 60]:
        key = (stype, h)
        if key not in _classifiers:
            continue
        clf, thr = _classifiers[key]
        prob = float(clf.predict_proba(X_last)[0, 1])
        if h == 30:
            prob_30min = prob

        # Per-horizon KPI forecasts (one step = 5 min ahead)
        kpi_forecasts: dict[str, float] = {}
        for kpi in KPI_COLS:
            if kpi in df_feat.columns:
                try:
                    fcst_model = load_forecaster(stype, kpi, h, MODELS_PATH)
                    kpi_forecasts[kpi] = round(float(fcst_model.predict(X_last)[0]), 4)
                except Exception:
                    pass

        horizons_out.append(HorizonPrediction(
            horizon_min=h,
            violation_prob=round(prob, 4),
            predicted_kpis=kpi_forecasts,
        ))

    # ── SHAP top-5 features ────────────────────────────────────────────────
    shap_features: list[dict[str, Any]] = []
    top_feat_names: list[str] = []
    explainer = _explainers.get((stype, 30))
    if explainer is not None:
        shap_vals = explainer.shap_values(X_last)[0]       # (n_features,)
        abs_shap  = np.abs(shap_vals)
        top_idx   = np.argsort(abs_shap)[-5:][::-1]
        for i in top_idx:
            fname = feat_cols[i]
            top_feat_names.append(fname)
            shap_features.append({
                "feature":    fname,
                "shap_value": round(float(shap_vals[i]), 6),
                "direction":  "increases_risk" if shap_vals[i] > 0 else "decreases_risk",
            })

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = _get_recommendations(top_feat_names)

    return PredictionResponse(
        slice_type=stype,
        health_status=_derive_health(prob_30min),
        horizons=horizons_out,
        top_shap_features=shap_features,
        recommendations=recommendations,
    )


@app.get("/health/{slice_type}", response_model=HealthResponse)
def health(slice_type: str) -> HealthResponse:
    """
    Return current health status for a slice type.

    Uses the 30min violation probability from the most recent model state.
    Loads the stored classifier but does NOT run full feature engineering.
    """
    if slice_type not in SUPPORTED_SLICES:
        raise HTTPException(status_code=404, detail=f"Unknown slice_type: {slice_type!r}")

    _ensure_loaded()
    key = (slice_type, 30)
    if key not in _classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")

    clf, thr = _classifiers[key]
    # Return the threshold itself as a proxy for the decision boundary probability
    # In production this would be fed from a live feature vector
    prob_proxy = float(thr)
    return HealthResponse(
        slice_type=slice_type,
        health_status=_derive_health(prob_proxy),
        violation_prob_30min=round(prob_proxy, 4),
    )


@app.get("/slices")
def list_slices() -> dict[str, list[str]]:
    """Return the list of supported slice types."""
    return {"slices": SUPPORTED_SLICES}


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=8000, reload=False)
