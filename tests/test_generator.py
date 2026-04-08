"""
Unit tests for the synthetic 5G slice data generator.

Requirements covered:
  2.1  — 90 days × 288 intervals = 25,920 rows
  2.8  — every row has a non-null event_type
  2.13 — any_breach positive rate between 2% and 10% (injection enabled)
  12.7 — event_type column present and non-null
  13.2 — breach rate within expected range for eMBB
  13.6 — no dl_throughput breach when injection disabled
"""

import pytest
import pandas as pd

from src.data.generator import generate_slice_data, INTERVALS_PER_DAY


# ─── 2.1: Row count ───────────────────────────────────────────────────────────

def test_embb_90day_row_count():
    """generate_slice_data('eMBB', days=90) must produce exactly 25,920 rows."""
    df = generate_slice_data("eMBB", days=90, seed=42)
    expected = 90 * INTERVALS_PER_DAY  # 25,920
    assert len(df) == expected, (
        f"Expected {expected} rows, got {len(df)}"
    )


# ─── 2.13 / 13.2: Breach rate ─────────────────────────────────────────────────

def test_embb_breach_rate_within_bounds():
    """
    any_breach positive rate must be between 2% and 10% for a 90-day eMBB run
    with violation injection enabled.

    Requirements: 2.13, 13.2
    """
    df = generate_slice_data("eMBB", days=90, inject_violations=True, seed=42)
    rate = df["any_breach"].mean()
    assert 0.02 <= rate <= 0.10, (
        f"any_breach rate {rate:.4f} is outside the expected [0.02, 0.10] range"
    )


# ─── 13.6: No throughput breach without injection ─────────────────────────────

def test_embb_no_throughput_breach_without_injection():
    """
    A 30-day eMBB run with inject_violations=False must have
    dl_throughput >= 45 Mbps for every row.

    Requirements: 13.6
    """
    df = generate_slice_data("eMBB", days=30, inject_violations=False, seed=42)
    min_thr = df["dl_throughput"].min()
    assert min_thr >= 45.0, (
        f"dl_throughput dropped to {min_thr:.2f} Mbps without injection "
        f"(expected >= 45 Mbps)"
    )


# ─── 2.8 / 12.7: event_type non-null ─────────────────────────────────────────

def test_every_row_has_non_null_event_type():
    """
    Every row in a 90-day eMBB run must have a non-null event_type column.

    Requirements: 2.8, 12.7
    """
    df = generate_slice_data("eMBB", days=90, seed=42)
    assert "event_type" in df.columns, "event_type column is missing"
    null_count = df["event_type"].isna().sum()
    assert null_count == 0, (
        f"Found {null_count} null values in event_type column"
    )
