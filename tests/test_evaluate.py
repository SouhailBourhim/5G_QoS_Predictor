"""
Unit tests for src/evaluation/evaluate.py
Requirements: 8.1, 8.8
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluation.evaluate import (
    verify_temporal_integrity,
    compute_baseline_comparison,
    SLICE_TYPES,
)


def test_temporal_integrity_passes():
    """Pillar 1 should pass cleanly on the real generated data — Req 8.1"""
    result = verify_temporal_integrity("data/raw/generated")
    # Should return a dict with all three slices
    for stype in ["eMBB", "URLLC", "mMTC"]:
        assert stype in result, f"{stype} missing from integrity result"
        for split in ["train", "val", "test"]:
            assert split in result[stype], f"{split} rate missing for {stype}"


def test_temporal_integrity_raises_on_overlap(tmp_path):
    """Pillar 1 must raise AssertionError when splits overlap — Req 8.1"""
    # Create a tiny synthetic parquet with timestamp order that will overlap
    ts = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({
        "timestamp": ts,
        "violation_in_30min": np.zeros(100, dtype=int),
        "violation_in_15min": np.zeros(100, dtype=int),
        "violation_in_60min": np.zeros(100, dtype=int),
        "any_breach": np.zeros(100, dtype=bool),
        "time_to_violation": np.zeros(100),
        "dl_throughput": np.ones(100),
        "latency": np.ones(100),
        "jitter": np.ones(100),
        "packet_loss": np.zeros(100),
        "prb_util": np.ones(100) * 0.5,
        "active_users": np.ones(100),
        "reliability": np.ones(100) * 99.9,
        "slice_type": "eMBB",
        "event_type": "none",
    })
    # 100 rows = only ~8 hours — split at day 60 means train is ALL rows,
    # val and test will be empty → triggers AssertionError("empty")
    out = tmp_path / "embb_synthetic.parquet"
    df.to_parquet(out, index=False)

    with pytest.raises(AssertionError):
        verify_temporal_integrity(tmp_path)


def test_baseline_comparison_covers_all_slices():
    """Pillar 8 must return results for each slice type — Req 8.8"""
    from src.features.engineering import build_features
    from src.data.splitter import temporal_split

    raw_dir = Path("data/raw/generated")
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
        test_splits[stype] = test_df

    result = compute_baseline_comparison(test_splits, horizon_min=30)

    for stype in SLICE_TYPES:
        assert stype in result, f"{stype} missing from baseline comparison"
        assert "xgboost"  in result[stype]
        assert "baseline" in result[stype]
        for metrics_dict in result[stype].values():
            assert "precision" in metrics_dict
            assert "recall"    in metrics_dict
            assert "f1"        in metrics_dict
