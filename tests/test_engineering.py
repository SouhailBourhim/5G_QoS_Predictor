import pytest
import pandas as pd
import numpy as np
from src.features.engineering import build_features, add_sla_proximity
from src.utils.config import SLICE_CONFIGS, SLAThreshold

@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "slice_type": ["eMBB"] * 10,
        "dl_throughput": [60.0] * 10,  # SLA min is 50 -> margin should be 10 (positive)
        "latency": [20.0] * 10,        # SLA max is 30 -> margin should be 10 (positive)
        "packet_loss": [0.5] * 10,
        "reliability": [99.0] * 10,
        "jitter": [5.0] * 10,
        "prb_util": [0.4] * 10,
        "active_users": [10.0] * 10
    })
    return df

def test_build_features_column_count(sample_df):
    others = {"URLLC": sample_df.copy(), "mMTC": sample_df.copy()}
    feats = build_features(sample_df, "eMBB", others)
    # Reqs: at least 200 columns for a valid input
    assert len(feats.columns) >= 200

def test_lag_step_names(sample_df):
    feats = build_features(sample_df, "eMBB")
    lag_cols = [c for c in feats.columns if "_lag" in c]
    
    assert len(lag_cols) > 0, "No lag columns found"
    
    for c in lag_cols:
        # Expected format: <kpi>_lag<N>
        lag_val = int(c.split("_lag")[-1])
        assert lag_val >= 1

def test_sla_margin_sign(sample_df):
    sla_thresholds = [
        SLAThreshold("dl_throughput_mbps", 50.0, "min", 15),
        SLAThreshold("latency_ms", 30.0, "max", 15)
    ]
    df_out = add_sla_proximity(sample_df, "eMBB", sla_thresholds)
    
    # dl_throughput is 60, min threshold is 50 -> margin = 10 > 0
    assert (df_out["dl_throughput_sla_margin"] == 10.0).all()
    # latency is 20, max threshold is 30 -> margin = 10 > 0
    assert (df_out["latency_sla_margin"] == 10.0).all()
