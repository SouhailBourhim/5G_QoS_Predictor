"""
Synthetic 5G Slice Data Generator

Produces 90 days × 288 five-minute intervals = 25,920 rows per slice.
All KPI distribution parameters are loaded from calibration_params.yaml.

Layers:
  1+2: Intraday load profile × weekly modulation × daily jitter
  3:   KPI derivation from load (physical relationships)
  4:   AR(1) autocorrelated noise from calibrated distributions
  5:   Event injection with linear degradation ramps
  6:   Cross-slice PRB coupling penalty
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# ─── PATHS ────────────────────────────────────────────────────────────────────

CALIBRATION_YAML = RAW_DATA_DIR / "5g_nidd" / "calibration_params.yaml"
GENERATED_DIR = RAW_DATA_DIR / "generated"

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

INTERVALS_PER_DAY = 288          # 24h × 12 intervals/h (5-min granularity)
MINUTES_PER_INTERVAL = 5

# Weekly modulation factors per slice
WEEKLY_FACTORS = {
    "eMBB":  {"weekday": 1.00, "weekend": 1.15},
    "URLLC": {"weekday": 1.00, "weekend": 0.30},
    "mMTC":  {"weekday": 1.00, "weekend": 0.95},
}

# KPI ranges used for load-to-KPI mapping (slice-specific where needed)
# Format: {kpi: (V_min, V_max)}
# Designed so that at max load (1.0), KPIs approach but don't breach SLA thresholds.
# Events push KPIs beyond these ranges to cause actual violations.
KPI_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "eMBB": {
        "dl_throughput": (60.0,  200.0),   # Mbps — min > SLA(50) so normal load never breaches
        "latency":       (2.0,   25.0),    # ms — max < SLA(30)
        "jitter":        (0.1,   3.0),     # ms
        "packet_loss":   (0.0,   0.7),     # % — max < SLA(1%)
        "prb_util":      (0.05,  0.55),    # fraction — kept low so total PRB rarely > 0.9
        "active_users":  (1.0,   100.0),   # count
        "reliability":   (97.0,  100.0),   # %
    },
    "URLLC": {
        "dl_throughput": (5.0,   50.0),
        "latency":       (0.5,   4.5),     # ms — max just below SLA(5), events push over
        "jitter":        (0.01,  0.85),    # ms — max below SLA(1), events push over
        "packet_loss":   (0.0,   0.05),    # %
        "prb_util":      (0.05,  0.50),    # fraction — kept low
        "active_users":  (1.0,   50.0),
        "reliability":   (99.9992, 100.0), # % — events push below SLA(99.999)
    },
    "mMTC": {
        "dl_throughput": (0.1,   10.0),
        "latency":       (10.0,  800.0),   # ms — max < SLA(1000)
        "jitter":        (1.0,   30.0),
        "packet_loss":   (0.0,   3.0),
        "prb_util":      (0.05,  0.40),    # fraction — kept low
        "active_users":  (100.0, 10000.0), # IoT devices
        "reliability":   (96.0,  100.0),   # % — min > SLA(95)
    },
}

# KPIs that INCREASE with load
LOAD_PROPORTIONAL_KPIS = ["latency", "jitter", "packet_loss", "prb_util", "active_users"]
# KPIs that DECREASE with load
LOAD_INVERSE_KPIS = ["dl_throughput", "reliability"]

# SLA thresholds for target construction
SLA_THRESHOLDS: dict[str, list[dict]] = {
    "eMBB": [
        {"kpi": "dl_throughput", "threshold": 50.0,  "direction": "min"},
        {"kpi": "latency",       "threshold": 30.0,  "direction": "max"},
        {"kpi": "packet_loss",   "threshold": 1.0,   "direction": "max"},
    ],
    "URLLC": [
        {"kpi": "latency",       "threshold": 5.0,    "direction": "max"},
        {"kpi": "reliability",   "threshold": 99.999, "direction": "min"},
        {"kpi": "jitter",        "threshold": 1.0,    "direction": "max"},
    ],
    "mMTC": [
        {"kpi": "reliability",   "threshold": 95.0,   "direction": "min"},  # delivery_rate
        {"kpi": "latency",       "threshold": 1000.0, "direction": "max"},
    ],
}

# Event definitions
EVENT_DEFINITIONS = {
    "traffic_surge":      {"slices": ["eMBB"],          "buildup_min": (5, 5),       "rate": 0.015},
    "gradual_congestion": {"slices": ["eMBB", "mMTC"],  "buildup_min": (30, 120),    "rate": 0.010},
    "interference":       {"slices": ["URLLC", "eMBB"], "buildup_min": (0, 0),       "rate": 0.002},
    "hw_degradation":     {"slices": ["eMBB", "URLLC", "mMTC"], "buildup_min": (60, 480), "rate": 0.002},
    "resource_starvation":{"slices": ["URLLC"],         "buildup_min": (15, 60),     "rate": 0.002},
    "iot_storm":          {"slices": ["mMTC"],           "buildup_min": (0, 0),       "rate": 0.012},
}

# Active window duration range (in intervals)
EVENT_ACTIVE_INTERVALS = (3, 12)   # 15 min – 1 h

# Degradation multiplier per slice — controls how strongly events push KPIs toward SLA breach
# Higher = more violations per event
DEGRADATION_MULTIPLIER = {
    "eMBB":  0.6,   # moderate — SLA thresholds are not too tight
    "URLLC": 0.8,   # high — need to push tight SLA KPIs over threshold
    "mMTC":  0.7,   # higher — mMTC SLA thresholds have more headroom
}


# ─── CALIBRATION LOADER ───────────────────────────────────────────────────────

def _load_calibration(path: Path = CALIBRATION_YAML) -> dict:
    """Load calibration_params.yaml and return as dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ─── LAYER 1+2: LOAD PROFILE ──────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _embb_intraday(t: np.ndarray) -> np.ndarray:
    """Triple-hump Gaussian: morning commute, lunch, evening streaming."""
    return (
        0.15 * np.exp(-((t - 8.5) ** 2) / (2 * 1.8 ** 2))
        + 0.30 * np.exp(-((t - 12.5) ** 2) / (2 * 1.5 ** 2))
        + 0.45 * np.exp(-((t - 20.5) ** 2) / (2 * 2.2 ** 2))
        + 0.08
    )


def _urllc_intraday(t: np.ndarray) -> np.ndarray:
    """Sigmoid business-hours plateau."""
    return _sigmoid(3.0 * (t - 7.5)) - _sigmoid(3.0 * (t - 17.5))


def _mmtc_intraday(t: np.ndarray) -> np.ndarray:
    """Periodic impulse bursts: delta=1h, epsilon=0.1h duty cycle."""
    delta = 1.0   # hours
    epsilon = 0.1  # hours
    return 0.15 + 0.7 * ((t % delta) < epsilon).astype(float)


def build_load_profile(
    slice_type: str,
    n_timesteps: int,
    rng: np.random.Generator,
    start_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer 1+2: intraday pattern × weekly modulation × daily jitter.

    Returns:
        load: shape (n_timesteps,) — final load in [0, 1]
        weekly_factor: shape (n_timesteps,) — weekly modulation per timestep
        timestamps: DatetimeIndex of length n_timesteps
    """
    # Build timestamp index (5-min intervals)
    timestamps = pd.date_range(start=start_date, periods=n_timesteps, freq="5min")

    # Hour-of-day as float (0–24)
    t_hour = timestamps.hour + timestamps.minute / 60.0
    t_arr = t_hour.to_numpy(dtype=float)

    # Intraday profile
    if slice_type == "eMBB":
        intraday = _embb_intraday(t_arr)
    elif slice_type == "URLLC":
        intraday = _urllc_intraday(t_arr)
    elif slice_type == "mMTC":
        intraday = _mmtc_intraday(t_arr)
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    # Clip intraday to [0, 1]
    intraday = np.clip(intraday, 0.0, 1.0)

    # Weekly modulation: weekday vs weekend
    wf = WEEKLY_FACTORS[slice_type]
    is_weekend = np.array(timestamps.dayofweek >= 5, dtype=float)
    weekly_base = np.where(is_weekend, wf["weekend"], wf["weekday"])

    # Per-day Normal(1.0, 0.05) jitter — one value per calendar day
    days_offset = np.array((timestamps - timestamps[0]).days, dtype=int)
    n_days = int(days_offset.max()) + 1
    daily_jitter = rng.normal(1.0, 0.05, size=n_days)
    daily_jitter = np.clip(daily_jitter, 0.80, 1.20)  # guard against extremes
    per_step_jitter = daily_jitter[days_offset]

    weekly_factor = weekly_base * per_step_jitter

    # Final load
    load = np.clip(intraday * weekly_factor, 0.0, 1.0)

    return load, weekly_factor, timestamps


# ─── LAYER 3: KPI DERIVATION ──────────────────────────────────────────────────

def derive_kpis_from_load(
    load: np.ndarray,
    weekly_factor: np.ndarray,
    slice_type: str,
) -> pd.DataFrame:
    """
    Layer 3: physical KPI relationships from load level.

    Proportional KPIs:  V = V_min + (V_max - V_min) * load * weekly_factor
    Inverse KPIs:       V = V_max - (V_max - V_min) * load * weekly_factor
    """
    ranges = KPI_RANGES[slice_type]
    result = {}

    for kpi, (v_min, v_max) in ranges.items():
        if kpi in LOAD_PROPORTIONAL_KPIS:
            v = v_min + (v_max - v_min) * load * weekly_factor
        else:  # LOAD_INVERSE_KPIS
            v = v_max - (v_max - v_min) * load * weekly_factor
        result[kpi] = np.clip(v, v_min, v_max)

    return pd.DataFrame(result)


# ─── LAYER 4: AR(1) AUTOCORRELATED NOISE ─────────────────────────────────────

def _sample_noise(kpi_name: str, n: int, calibration: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n noise samples from the KPI-specific distribution in calibration_params.yaml.
    Returns noise scaled to a small fraction of the KPI's std.
    """
    # Map generator KPI names to calibration keys
    cal_key_map = {
        "dl_throughput": "dl_throughput",
        "latency":       "latency",
        "jitter":        "jitter",
        "packet_loss":   "packet_loss",
        "prb_util":      "packet_loss",   # use packet_loss distribution as proxy
        "active_users":  "active_users",
        "reliability":   "reliability",
    }
    cal_key = cal_key_map.get(kpi_name, "latency")
    params = calibration.get(cal_key, {})

    dist = params.get("distribution", "normal")
    std = float(params.get("std", 1.0))
    noise_scale = max(std * 0.05, 1e-6)  # 5% of empirical std as noise amplitude

    if dist == "lognormal":
        fp = params.get("fit_params", {})
        s = float(fp.get("s", 1.0))
        scale = float(fp.get("scale", 1.0))
        raw = rng.lognormal(mean=np.log(max(scale, 1e-9)), sigma=s, size=n)
        # Centre around zero
        noise = (raw - raw.mean()) * noise_scale / max(raw.std(), 1e-9)
    elif dist == "gamma":
        fp = params.get("fit_params", {})
        a = float(fp.get("a", 1.0))
        raw = rng.gamma(shape=max(a, 0.01), scale=1.0, size=n)
        noise = (raw - raw.mean()) * noise_scale / max(raw.std(), 1e-9)
    elif dist == "exponential":
        raw = rng.exponential(scale=1.0, size=n)
        noise = (raw - raw.mean()) * noise_scale / max(raw.std(), 1e-9)
    elif dist == "beta":
        fp = params.get("fit_params", {})
        a = float(fp.get("a", 0.5))
        b = float(fp.get("b", 0.5))
        raw = rng.beta(max(a, 0.01), max(b, 0.01), size=n)
        noise = (raw - 0.5) * noise_scale
    else:  # normal
        noise = rng.normal(0.0, noise_scale, size=n)

    return noise


def apply_autocorrelated_noise(
    kpi_series: np.ndarray,
    kpi_name: str,
    calibration: dict,
    rng: np.random.Generator,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Layer 4: AR(1) smoothing with alpha=0.3.

    noise[t] = alpha * noise[t-1] + (1 - alpha) * epsilon[t]
    """
    n = len(kpi_series)
    epsilon = _sample_noise(kpi_name, n, calibration, rng)

    ar_noise = np.zeros(n)
    ar_noise[0] = epsilon[0]
    for i in range(1, n):
        ar_noise[i] = alpha * ar_noise[i - 1] + (1.0 - alpha) * epsilon[i]

    return kpi_series + ar_noise


# ─── LAYER 5: EVENT INJECTION ─────────────────────────────────────────────────

def _build_event_schedule(
    slice_type: str,
    n_timesteps: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Build a list of event dicts for the given slice.
    Each event: {event_type, start_idx, buildup_steps, active_steps, severity}
    """
    schedule = []
    for event_type, defn in EVENT_DEFINITIONS.items():
        if slice_type not in defn["slices"]:
            continue

        rate = defn["rate"]  # expected events per interval
        # Poisson process: expected total events
        expected_events = int(round(n_timesteps * rate))
        n_events = max(1, rng.poisson(expected_events))

        # Spread events roughly uniformly with some randomness
        if n_events > 0:
            positions = rng.choice(n_timesteps, size=n_events, replace=False)
            positions.sort()

        buildup_range = defn["buildup_min"]
        for pos in positions:
            buildup_steps = 0
            if buildup_range[1] > 0:
                buildup_min = rng.integers(buildup_range[0], buildup_range[1] + 1)
                buildup_steps = int(buildup_min / MINUTES_PER_INTERVAL)

            active_steps = int(rng.integers(EVENT_ACTIVE_INTERVALS[0], EVENT_ACTIVE_INTERVALS[1] + 1))
            severity = float(rng.uniform(0.3, 1.0))

            # Ensure event fits within the timeline
            total_len = buildup_steps + active_steps
            if pos + total_len > n_timesteps:
                pos = max(0, n_timesteps - total_len - 1)

            schedule.append({
                "event_type": event_type,
                "start_idx": int(pos),
                "buildup_steps": buildup_steps,
                "active_steps": active_steps,
                "severity": severity,
            })

    return schedule


def inject_events(
    df: pd.DataFrame,
    slice_type: str,
    event_schedule: list[dict],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Layer 5: apply degradation ramps for each event type.

    During buildup:      degradation = severity × 0.7 × progress
    During active window: degradation = severity × Normal(1.0, 0.1)

    Degradation is applied as a multiplicative penalty to load-sensitive KPIs.
    """
    df = df.copy()
    n = len(df)

    # event_type column: default "normal"
    event_col = np.full(n, "normal", dtype=object)

    ranges = KPI_RANGES[slice_type]

    for event in event_schedule:
        etype = event["event_type"]
        s_idx = event["start_idx"]
        buildup = event["buildup_steps"]
        active = event["active_steps"]
        severity = event["severity"]

        # Buildup phase
        for step in range(buildup):
            idx = s_idx + step
            if idx >= n:
                break
            progress = (step + 1) / max(buildup, 1)
            degradation = severity * 0.7 * progress
            _apply_degradation(df, idx, degradation, slice_type, ranges)
            event_col[idx] = etype

        # Active phase
        active_start = s_idx + buildup
        for step in range(active):
            idx = active_start + step
            if idx >= n:
                break
            degradation = severity * float(rng.normal(1.0, 0.1))
            degradation = max(0.0, degradation)
            _apply_degradation(df, idx, degradation, slice_type, ranges)
            event_col[idx] = etype

    df["event_type"] = event_col
    return df


def _apply_degradation(
    df: pd.DataFrame,
    idx: int,
    degradation: float,
    slice_type: str,
    ranges: dict[str, tuple[float, float]],
) -> None:
    """Apply degradation factor in-place at row idx."""
    mult = DEGRADATION_MULTIPLIER.get(slice_type, 0.4)

    # Degrade proportional KPIs upward (worse = higher)
    # Allow values above v_max so events can push past SLA thresholds
    for kpi in LOAD_PROPORTIONAL_KPIS:
        if kpi not in df.columns:
            continue
        v_min, v_max = ranges[kpi]
        span = v_max - v_min
        new_val = df.at[idx, kpi] + degradation * span * mult
        # PRB must stay in [0, 1] for cross-slice coupling to work correctly
        if kpi == "prb_util":
            new_val = min(1.0, new_val)
        df.at[idx, kpi] = new_val

    # Degrade inverse KPIs downward (worse = lower)
    # Allow values below v_min so events can push past SLA thresholds
    for kpi in LOAD_INVERSE_KPIS:
        if kpi not in df.columns:
            continue
        v_min, v_max = ranges[kpi]
        span = v_max - v_min
        df.at[idx, kpi] = df.at[idx, kpi] - degradation * span * mult


# ─── LAYER 6: CROSS-SLICE PRB COUPLING ───────────────────────────────────────

def apply_cross_slice_coupling(
    embb_df: pd.DataFrame,
    urllc_df: pd.DataFrame,
    mmtc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Layer 6: when total PRB > 90%, apply excess as penalty to eMBB throughput
    and latency with multiplier 1.5.
    """
    embb_df = embb_df.copy()

    total_prb = embb_df["prb_util"] + urllc_df["prb_util"] + mmtc_df["prb_util"]
    excess = np.maximum(0.0, total_prb - 0.90)

    # Penalty on eMBB dl_throughput (reduce) — allow below normal range min
    v_min_thr, v_max_thr = KPI_RANGES["eMBB"]["dl_throughput"]
    thr_penalty = excess * 1.5 * (v_max_thr - v_min_thr)
    embb_df["dl_throughput"] = np.clip(
        embb_df["dl_throughput"] - thr_penalty,
        0.0, v_max_thr,  # allow down to 0 so coupling can cause breaches
    )

    # Penalty on eMBB latency (increase) — allow above normal range max
    v_min_lat, v_max_lat = KPI_RANGES["eMBB"]["latency"]
    lat_penalty = excess * 1.5 * (v_max_lat - v_min_lat)
    embb_df["latency"] = np.clip(
        embb_df["latency"] + lat_penalty,
        v_min_lat, v_max_lat * 3.0,  # allow up to 3× max
    )

    return embb_df, urllc_df, mmtc_df


# ─── TARGET CONSTRUCTION ──────────────────────────────────────────────────────

def _is_breach(df: pd.DataFrame, slice_type: str) -> np.ndarray:
    """Return boolean array: True where any SLA threshold is breached."""
    thresholds = SLA_THRESHOLDS[slice_type]
    breach = np.zeros(len(df), dtype=bool)
    for t in thresholds:
        kpi = t["kpi"]
        if kpi not in df.columns:
            continue
        if t["direction"] == "max":
            breach |= (df[kpi].to_numpy() > t["threshold"])
        else:
            breach |= (df[kpi].to_numpy() < t["threshold"])
    return breach


def build_targets(df: pd.DataFrame, slice_type: str) -> pd.DataFrame:
    """
    Compute:
      any_breach              — bool, current timestep SLA breach
      time_to_violation       — minutes until next breach (sentinel=9999)
      violation_in_15min      — 1 if breach within next 3 intervals
      violation_in_30min      — 1 if breach within next 6 intervals
      violation_in_60min      — 1 if breach within next 12 intervals
    """
    df = df.copy()
    breach = _is_breach(df, slice_type)
    df["any_breach"] = breach

    n = len(df)
    time_to_viol = np.full(n, 9999.0)
    viol_15 = np.zeros(n, dtype=int)
    viol_30 = np.zeros(n, dtype=int)
    viol_60 = np.zeros(n, dtype=int)

    horizons = {
        "15min": 3,   # 3 × 5 min = 15 min
        "30min": 6,
        "60min": 12,
    }

    for i in range(n):
        # time_to_violation: scan forward for next breach
        for j in range(i + 1, min(i + horizons["60min"] + 1, n)):
            if breach[j]:
                time_to_viol[i] = (j - i) * MINUTES_PER_INTERVAL
                break

        # horizon targets
        for label, steps in horizons.items():
            window = breach[i + 1 : i + 1 + steps]
            if len(window) > 0 and window.any():
                if label == "15min":
                    viol_15[i] = 1
                elif label == "30min":
                    viol_30[i] = 1
                else:
                    viol_60[i] = 1

    df["time_to_violation"] = time_to_viol
    df["violation_in_15min"] = viol_15
    df["violation_in_30min"] = viol_30
    df["violation_in_60min"] = viol_60

    return df


# ─── TOP-LEVEL ENTRY POINTS ───────────────────────────────────────────────────

def generate_slice_data(
    slice_type: str,
    days: int = 90,
    inject_violations: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Top-level entry point. Returns DataFrame with all KPIs + targets.

    Args:
        slice_type: 'eMBB', 'URLLC', or 'mMTC'
        days: number of days to generate
        inject_violations: whether to inject events that cause SLA violations
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    calibration = _load_calibration()

    n_timesteps = days * INTERVALS_PER_DAY

    # Layer 1+2: load profile
    load, weekly_factor, timestamps = build_load_profile(slice_type, n_timesteps, rng)

    # Layer 3: KPI derivation
    df = derive_kpis_from_load(load, weekly_factor, slice_type)

    # Layer 4: AR(1) noise
    for kpi in df.columns:
        df[kpi] = apply_autocorrelated_noise(
            df[kpi].to_numpy(), kpi, calibration, rng, alpha=0.3
        )
        # Re-clip after noise
        v_min, v_max = KPI_RANGES[slice_type][kpi]
        df[kpi] = np.clip(df[kpi], v_min, v_max)

    # Add timestamp and slice metadata
    df["timestamp"] = timestamps
    df["slice_type"] = slice_type

    # Layer 5: event injection
    if inject_violations:
        event_schedule = _build_event_schedule(slice_type, n_timesteps, rng)
        df = inject_events(df, slice_type, event_schedule, rng)
    else:
        df["event_type"] = "normal"

    # Build targets
    df = build_targets(df, slice_type)

    # Reorder columns
    meta_cols = ["timestamp", "slice_type", "event_type"]
    kpi_cols = list(KPI_RANGES[slice_type].keys())
    target_cols = ["any_breach", "time_to_violation", "violation_in_15min",
                   "violation_in_30min", "violation_in_60min"]
    df = df[meta_cols + kpi_cols + target_cols]

    logger.info(
        "%s: %d rows, any_breach rate=%.3f",
        slice_type, len(df), df["any_breach"].mean(),
    )
    return df


def generate_all_slices(days: int = 90) -> dict[str, pd.DataFrame]:
    """
    Generate eMBB, URLLC, mMTC, apply cross-slice coupling, save Parquet.

    Returns dict mapping slice_type -> DataFrame.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Generate each slice independently (different seeds for variety)
    slices = {}
    seeds = {"eMBB": 42, "URLLC": 43, "mMTC": 44}
    for slice_type, seed in seeds.items():
        logger.info("Generating %s slice (seed=%d)…", slice_type, seed)
        slices[slice_type] = generate_slice_data(slice_type, days=days, seed=seed)

    # Layer 6: cross-slice PRB coupling (modifies eMBB in-place)
    slices["eMBB"], slices["URLLC"], slices["mMTC"] = apply_cross_slice_coupling(
        slices["eMBB"], slices["URLLC"], slices["mMTC"]
    )

    # Rebuild targets for eMBB after coupling (PRB penalty may cause new breaches)
    slices["eMBB"] = build_targets(
        slices["eMBB"].drop(
            columns=["any_breach", "time_to_violation",
                     "violation_in_15min", "violation_in_30min", "violation_in_60min"],
            errors="ignore",
        ),
        "eMBB",
    )

    # Save Parquet
    for slice_type, df in slices.items():
        out_path = GENERATED_DIR / f"{slice_type.lower()}_synthetic.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s → %s (%d rows)", slice_type, out_path, len(df))

    return slices


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    result = generate_all_slices()
    for st, df in result.items():
        breach_rate = df["any_breach"].mean()
        print(f"{st}: {len(df)} rows, any_breach={breach_rate:.3%}")
