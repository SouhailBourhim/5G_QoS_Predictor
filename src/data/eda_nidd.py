"""
EDA Module for 5G-NIDD Dataset

Extracts calibration parameters from real 5G-NIDD Argus flow records.
Outputs calibration_params.yaml and diagnostic plots.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.data.argus_loader import ArgusFlowLoader
from src.utils.config import RAW_DATA_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

# KPI columns produced by ArgusFlowLoader.compute_kpis()
KPI_COLS = [
    "dl_throughput_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "reliability_pct",
]

# Distribution to fit per KPI
KPI_DISTRIBUTIONS = {
    "dl_throughput_mbps": "lognormal",
    "ul_throughput_mbps": "lognormal",
    "latency_ms": "gamma",
    "jitter_ms": "exponential",
    "packet_loss_pct": "beta",
    "reliability_pct": "normal",
    "active_users": "normal",
}

# Mobility scenario label mapping (based on Label column in 5G-NIDD)
MOBILITY_SCENARIOS = {
    "vehicular": ["vehicular", "vehicle", "car"],
    "pedestrian": ["pedestrian", "walk", "ped"],
    "static": ["static", "stationary", "fixed"],
}

CALIBRATION_YAML_PATH = RAW_DATA_DIR / "5g_nidd" / "calibration_params.yaml"
FIGURES_DIR = REPORTS_DIR / "figures"


# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def compute_kpi_stats(series: pd.Series, kpi_name: str) -> dict:
    """
    Fit a distribution to a KPI series and return summary statistics.

    Args:
        series: 1-D numeric series (NaNs dropped internally).
        kpi_name: Name used to look up the target distribution family.

    Returns:
        dict with keys: distribution, mean, std, median, min, max, range.
    """
    clean = series.dropna()
    if len(clean) == 0:
        logger.warning(f"No valid data for KPI '{kpi_name}' — returning zeros.")
        return {
            "distribution": KPI_DISTRIBUTIONS.get(kpi_name, "normal"),
            "mean": 0.0, "std": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "range": [0.0, 0.0],
        }

    dist_name = KPI_DISTRIBUTIONS.get(kpi_name, "normal")

    # Fit distribution parameters (stored for reference; generator uses mean/std)
    try:
        if dist_name == "lognormal":
            # scipy lognormal: shape=s, loc, scale=exp(mu)
            s, loc, scale = stats.lognorm.fit(clean.clip(lower=1e-9), floc=0)
            fit_params = {"s": float(s), "loc": float(loc), "scale": float(scale)}
        elif dist_name == "gamma":
            # Gamma requires strictly positive values; clip zeros to a small epsilon
            pos = clean.clip(lower=1e-9)
            pos = pos[pos > 0]
            if len(pos) == 0:
                fit_params = {}
            else:
                a, loc, scale = stats.gamma.fit(pos, floc=0)
                fit_params = {"a": float(a), "loc": float(loc), "scale": float(scale)}
        elif dist_name == "exponential":
            loc, scale = stats.expon.fit(clean.clip(lower=0), floc=0)
            fit_params = {"loc": float(loc), "scale": float(scale)}
        elif dist_name == "beta":
            # Beta requires data in (0, 1); packet_loss_pct is 0–100
            normed = (clean.clip(0, 100) / 100).clip(1e-6, 1 - 1e-6)
            a, b, loc, scale = stats.beta.fit(normed, floc=0, fscale=1)
            fit_params = {"a": float(a), "b": float(b)}
        else:
            fit_params = {}
    except Exception as exc:
        logger.warning(f"Distribution fit failed for '{kpi_name}': {exc}")
        fit_params = {}

    return {
        "distribution": dist_name,
        "fit_params": fit_params,
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "range": [float(clean.min()), float(clean.max())],
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix across all numeric KPI columns.

    Args:
        df: DataFrame containing at least the KPI columns.

    Returns:
        Correlation matrix as a DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr(method="pearson")
    logger.info("Pearson correlation matrix computed (%d × %d)", len(numeric_cols), len(numeric_cols))
    return corr


def compute_mobility_variance(df: pd.DataFrame) -> dict:
    """
    Compute per-KPI variance for each mobility scenario.

    The 5G-NIDD dataset does not have an explicit mobility column, so we
    proxy mobility by grouping on the 'label' column when present, or fall
    back to synthetic scenario assignment based on row index thirds.

    Args:
        df: Flow-level DataFrame with KPI columns.

    Returns:
        Nested dict: {scenario: {kpi: variance}}.
    """
    kpi_cols = [c for c in KPI_COLS if c in df.columns]
    result: dict = {s: {} for s in MOBILITY_SCENARIOS}

    # Try to map existing labels to mobility scenarios
    if "label" in df.columns:
        label_lower = df["label"].str.lower().fillna("")
        scenario_mask = {}
        for scenario, keywords in MOBILITY_SCENARIOS.items():
            mask = label_lower.apply(lambda x: any(k in x for k in keywords))
            scenario_mask[scenario] = mask

        any_matched = sum(m.sum() for m in scenario_mask.values())
    else:
        any_matched = 0

    if any_matched == 0:
        # Fallback: split rows into thirds as proxy for mobility scenarios
        logger.info("No mobility labels found — using row-thirds as proxy scenarios.")
        n = len(df)
        thirds = [
            df.iloc[: n // 3],
            df.iloc[n // 3 : 2 * n // 3],
            df.iloc[2 * n // 3 :],
        ]
        for scenario, subset in zip(MOBILITY_SCENARIOS.keys(), thirds):
            for kpi in kpi_cols:
                result[scenario][kpi] = float(subset[kpi].dropna().var()) if kpi in subset else 0.0
    else:
        for scenario, mask in scenario_mask.items():
            subset = df[mask]
            for kpi in kpi_cols:
                result[scenario][kpi] = float(subset[kpi].dropna().var()) if len(subset) > 0 else 0.0

    # Log vehicular vs static comparison
    for kpi in kpi_cols:
        v_var = result["vehicular"].get(kpi, 0.0)
        s_var = result["static"].get(kpi, 0.0)
        if v_var >= s_var:
            logger.info("✓ %s: vehicular variance (%.4f) >= static variance (%.4f)", kpi, v_var, s_var)
        else:
            logger.warning("✗ %s: vehicular variance (%.4f) < static variance (%.4f) — proxy data limitation", kpi, v_var, s_var)

    return result


def save_calibration_params(params: dict, path: Path) -> None:
    """
    Write calibration parameters to a YAML file importable by the generator.

    Args:
        params: Nested dict of KPI → stats.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to plain Python for YAML serialisation
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    clean_params = _to_python(params)

    with open(path, "w") as fh:
        yaml.dump(clean_params, fh, default_flow_style=False, sort_keys=True)

    logger.info("Calibration params saved → %s", path)


def save_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Save a time-series overview plot and per-KPI distribution histograms.

    Args:
        df: Aggregated time-series DataFrame (output of aggregate_to_timeseries).
        figures_dir: Directory to write PNG files.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # aggregate_to_timeseries produces suffixed columns (e.g. dl_throughput_mbps_mean).
    # Accept both exact matches and columns that start with a KPI name.
    kpi_cols = []
    for kpi in KPI_COLS:
        if kpi in df.columns:
            kpi_cols.append(kpi)
        else:
            # Pick the first suffixed variant (prefer _mean)
            candidates = [c for c in df.columns if c.startswith(kpi + "_")]
            if candidates:
                preferred = next((c for c in candidates if c.endswith("_mean")), candidates[0])
                kpi_cols.append(preferred)

    if not kpi_cols:
        logger.warning("No KPI columns found in time-series DataFrame — skipping plots.")
        return

    # ── Time-series overview ──────────────────────────────────────────────────
    ts_col = "timestamp" if "timestamp" in df.columns else df.index.name
    n_kpis = len(kpi_cols)
    fig, axes = plt.subplots(n_kpis, 1, figsize=(14, 3 * n_kpis), sharex=True)
    if n_kpis == 1:
        axes = [axes]

    x = df[ts_col] if ts_col and ts_col in df.columns else df.index

    for ax, kpi in zip(axes, kpi_cols):
        ax.plot(x, df[kpi], linewidth=0.8, alpha=0.85)
        ax.set_ylabel(kpi, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle("5G-NIDD KPI Time-Series Overview", fontsize=12, y=1.01)
    fig.tight_layout()
    out = figures_dir / "timeseries_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time-series overview → %s", out)

    # ── Per-KPI distribution histograms ──────────────────────────────────────
    cols_per_row = 3
    n_rows = (n_kpis + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, kpi in zip(axes_flat, kpi_cols):
        clean = df[kpi].dropna()
        ax.hist(clean, bins=50, edgecolor="none", alpha=0.75)
        ax.set_title(kpi, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n_kpis:]:
        ax.set_visible(False)

    fig.suptitle("5G-NIDD KPI Distributions", fontsize=12)
    fig.tight_layout()
    out = figures_dir / "kpi_distributions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved KPI distributions → %s", out)


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def run_eda(
    data_dir: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> dict:
    """
    Full EDA pipeline: load 5G-NIDD → compute stats → save YAML + plots.

    Args:
        data_dir: Directory containing 5G-NIDD CSV files.
        calibration_path: Output path for calibration_params.yaml.
        figures_dir: Output directory for plots.
        nrows: Limit rows loaded (useful for quick runs).

    Returns:
        calibration_params dict (same content written to YAML).
    """
    data_dir = data_dir or (RAW_DATA_DIR / "5g_nidd")
    calibration_path = calibration_path or CALIBRATION_YAML_PATH
    figures_dir = figures_dir or FIGURES_DIR

    # ── 1. Load data ──────────────────────────────────────────────────────────
    loader = ArgusFlowLoader(data_dir=data_dir)

    # Prefer the combined BTS file; fall back to any CSV in the directory
    combined = data_dir / "bts_combined" / "BTS1_BTS2_fields_preserved.csv"
    file_path = combined if combined.exists() else None

    logger.info("Loading 5G-NIDD flows (nrows=%s)…", nrows)
    df_flows, df_ts = loader.load_and_process(
        file_path=file_path,
        nrows=nrows,
        filter_normal=False,   # keep all traffic for distribution fitting
        aggregate_window="5min",
    )

    # ── 2. Per-KPI statistics ─────────────────────────────────────────────────
    calibration_params: dict = {}

    # Flow-level KPIs (richer statistics from individual flows)
    flow_kpi_map = {
        "dl_throughput": "dl_throughput_mbps",
        "ul_throughput": "ul_throughput_mbps",
        "latency": "latency_ms",
        "jitter": "jitter_ms",
        "packet_loss": "packet_loss_pct",
        "reliability": "reliability_pct",
    }

    for param_key, col in flow_kpi_map.items():
        if col in df_flows.columns:
            calibration_params[param_key] = compute_kpi_stats(df_flows[col], col)
        else:
            logger.warning("Column '%s' not found in flow data — skipping.", col)

    # Active users from time-series aggregation
    if "active_users" in df_ts.columns:
        calibration_params["active_users"] = compute_kpi_stats(df_ts["active_users"], "active_users")

    # ── 3. Correlations ───────────────────────────────────────────────────────
    corr_cols = [c for c in flow_kpi_map.values() if c in df_flows.columns]
    if corr_cols:
        corr_matrix = compute_correlations(df_flows[corr_cols])
        # Log key correlations mentioned in requirements
        if "dl_throughput_mbps" in corr_matrix and "latency_ms" in corr_matrix:
            logger.info(
                "Correlation dl_throughput ↔ latency: %.3f",
                corr_matrix.loc["dl_throughput_mbps", "latency_ms"],
            )
        calibration_params["_correlations"] = corr_matrix.to_dict()

    # ── 4. Mobility variance ──────────────────────────────────────────────────
    mobility_var = compute_mobility_variance(df_flows)
    calibration_params["_mobility_variance"] = mobility_var

    # ── 5. Save YAML ──────────────────────────────────────────────────────────
    # Strip internal keys (prefixed with _) before writing
    public_params = {k: v for k, v in calibration_params.items() if not k.startswith("_")}
    save_calibration_params(public_params, calibration_path)

    # ── 6. Save plots ─────────────────────────────────────────────────────────
    save_plots(df_ts, figures_dir)

    logger.info("EDA complete.")
    return calibration_params


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    params = run_eda()
    print("\nCalibration parameters written to:", CALIBRATION_YAML_PATH)
    for kpi, stats_dict in params.items():
        if not kpi.startswith("_"):
            print(f"  {kpi}: mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}")
