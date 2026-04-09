"""
Unit tests for src/data/splitter.py — temporal split integrity.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.splitter import temporal_split


@pytest.fixture
def synthetic_90day_df():
    """90-day eMBB-style DataFrame at 5-minute granularity."""
    timestamps = pd.date_range("2024-01-01", periods=90 * 288, freq="5min")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "dl_throughput": rng.uniform(50, 120, len(timestamps)),
        "latency": rng.uniform(5, 30, len(timestamps)),
        "violation_in_30min": rng.integers(0, 2, len(timestamps)),
    })
    return df


def test_no_timestamp_overlap(synthetic_90day_df):
    """Requirements 4.3, 4.4, 13.1 — no timestamp overlap between any two splits."""
    train, val, test = temporal_split(synthetic_90day_df)

    # train / val boundary
    assert train["timestamp"].max() < val["timestamp"].min(), \
        "train and val overlap!"

    # val / test boundary
    assert val["timestamp"].max() < test["timestamp"].min(), \
        "val and test overlap!"


def test_split_sizes(synthetic_90day_df):
    """Train=60d, Val=15d, Test=15d — each split should be non-empty and correct size."""
    train, val, test = temporal_split(synthetic_90day_df)

    # 60 days × 288 intervals
    assert len(train) == 60 * 288, f"Expected {60*288} train rows, got {len(train)}"
    assert len(val) == 15 * 288, f"Expected {15*288} val rows, got {len(val)}"
    assert len(test) == 15 * 288, f"Expected {15*288} test rows, got {len(test)}"


def test_chronological_order(synthetic_90day_df):
    """All splits must be sorted chronologically (no shuffle)."""
    train, val, test = temporal_split(synthetic_90day_df)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        assert split["timestamp"].is_monotonic_increasing, \
            f"{name} split is not in chronological order!"


def test_total_rows_preserved(synthetic_90day_df):
    """No rows dropped or duplicated during splitting."""
    train, val, test = temporal_split(synthetic_90day_df)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_90day_df), \
        f"Row count mismatch: {total} != {len(synthetic_90day_df)}"
