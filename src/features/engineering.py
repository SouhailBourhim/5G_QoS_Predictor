import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import SLICE_CONFIGS

def add_lag_features(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    """Category 1: lags [1,3,6,12,24,36,72,144]"""
    lags = [1, 3, 6, 12, 24, 36, 72, 144]
    out_cols = {}
    for col in kpi_cols:
        for lag in lags:
            out_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_rolling_stats(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    """Category 2: windows [6,12,36,72,144,288] × {mean,std,range,cv}"""
    windows = [6, 12, 36, 72, 144, 288]
    out_cols = {}
    for col in kpi_cols:
        for win in windows:
            roll = df[col].rolling(window=win, min_periods=1)
            mean = roll.mean()
            std = roll.std().fillna(0)
            target_min = roll.min()
            target_max = roll.max()
            rng = target_max - target_min
            
            cv = np.where(mean != 0, std / mean, 0)
            
            out_cols[f"{col}_roll{win}_mean"] = mean
            out_cols[f"{col}_roll{win}_std"] = std
            out_cols[f"{col}_roll{win}_range"] = rng
            out_cols[f"{col}_roll{win}_cv"] = cv
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_ewma_features(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    """Category 3: spans [6,12,36]"""
    spans = [6, 12, 36]
    out_cols = {}
    for col in kpi_cols:
        for span in spans:
            out_cols[f"{col}_ewma{span}"] = df[col].ewm(span=span, min_periods=1).mean()
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_rate_of_change(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    """Category 4: diff1, diff6, diff2, trend_sign"""
    out_cols = {}
    for col in kpi_cols:
        diff1 = df[col].diff(1).fillna(0)
        diff6 = df[col].diff(6).fillna(0)
        diff2 = diff1.diff(1).fillna(0)
        trend_sign = np.sign(diff1).rolling(window=6, min_periods=1).mean()
        
        out_cols[f"{col}_diff1"] = diff1
        out_cols[f"{col}_diff6"] = diff6
        out_cols[f"{col}_diff2"] = diff2
        out_cols[f"{col}_trend_sign"] = trend_sign
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Category 5: hour_sin/cos, dow_sin/cos, is_weekend, etc."""
    ts = df['timestamp']
    out_cols = {}
    
    hours = ts.dt.hour + ts.dt.minute / 60.0
    out_cols['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    out_cols['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    dows = ts.dt.dayofweek
    out_cols['dow_sin'] = np.sin(2 * np.pi * dows / 7)
    out_cols['dow_cos'] = np.cos(2 * np.pi * dows / 7)
    
    out_cols['is_weekend'] = (dows >= 5).astype(int)
    
    is_biz = ((hours >= 8) & (hours < 18)).astype(int)
    is_peak = ((hours >= 18) & (hours < 23)).astype(int)
    is_off = ((hours < 8) | (hours >= 23)).astype(int)
    
    out_cols['is_business_hours'] = is_biz
    out_cols['is_peak_evening'] = is_peak
    out_cols['is_off_peak'] = is_off
    
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_sla_proximity(df: pd.DataFrame, slice_type: str, sla_thresholds: list) -> pd.DataFrame:
    """Category 6: sla_margin, sla_margin_norm, rolling min margins, time_to_breach"""
    out_cols = {}
    
    def map_col(name: str):
        mapping = {
            "dl_throughput_mbps": "dl_throughput",
            "latency_ms": "latency",
            "packet_loss_pct": "packet_loss",
            "reliability_pct": "reliability",
            "jitter_ms": "jitter",
            "delivery_rate_pct": "reliability",
            "avg_latency_ms": "latency"
        }
        return mapping.get(name, name)
        
    for sla in sla_thresholds:
        kpi = map_col(sla.kpi_name)
        if kpi not in df.columns:
            continue
            
        thresh = sla.threshold
        val = df[kpi]
        
        if sla.direction == "min":
            margin = val - thresh
        else:
            margin = thresh - val
            
        margin_norm = margin / (thresh if thresh != 0 else 1)
        
        out_cols[f"{kpi}_sla_margin"] = margin
        out_cols[f"{kpi}_sla_margin_norm"] = margin_norm
        
        for win in [6, 12, 36]:
            out_cols[f"{kpi}_sla_margin_roll{win}_min"] = margin.rolling(window=win, min_periods=1).min()
            
        # time to breach based on diff
        # diff refers to change in margin
        # if diff < 0, margin is closing
        diff1 = margin.diff(1).fillna(0)
        ttb = np.where((diff1 < 0) & (margin > 0), 5.0 * margin / (-diff1), 999)
        ttb = np.clip(ttb, 0, 999)
        out_cols[f"{kpi}_time_to_breach"] = ttb
        
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_cross_kpi_features(df: pd.DataFrame, slice_type: str) -> pd.DataFrame:
    """Category 7: BDP, spectral_eff, eff_throughput, jitter_ratio, bdp_diff (eMBB only)"""
    if slice_type != "eMBB":
        return df
        
    out_cols = {}
    bw = df["dl_throughput"] * 1e6
    delay = df["latency"] / 1000.0
    bdp = bw * delay
    
    out_cols["cross_bdp"] = bdp
    out_cols["cross_spectral_eff"] = np.where(df["prb_util"] > 0, df["dl_throughput"] / df["prb_util"], 0)
    out_cols["cross_eff_throughput"] = df["dl_throughput"] * (df["reliability"] / 100.0 if "reliability" in df.columns else (100 - df["packet_loss"]) / 100.0)
    out_cols["cross_jitter_ratio"] = np.where(df["latency"] > 0, df["jitter"] / df["latency"], 0)
    out_cols["cross_bdp_diff"] = bdp.diff(6).fillna(0)
    
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def add_cross_slice_features(df: pd.DataFrame, slice_type: str, other_slices: dict) -> pd.DataFrame:
    """Category 8: competitor PRB, total PRB, rolling PRB means, active users/devices"""
    if not other_slices:
        return df
    
    out_cols = {}
    comp_prb = pd.Series(0.0, index=df.index)
    total_prb = df["prb_util"].copy()
    
    for s_name, s_df in other_slices.items():
        comp_prb += s_df["prb_util"]
        total_prb += s_df["prb_util"]
        
    out_cols["cross_competitor_prb"] = comp_prb
    out_cols["cross_total_prb"] = total_prb
    
    embb_df = other_slices.get("eMBB")
    if slice_type == "eMBB":
        embb_df = df
        
    if embb_df is not None:
        for win in [6, 12, 36]:
            out_cols[f"cross_embb_prb_roll{win}_mean"] = embb_df["prb_util"].rolling(window=win, min_periods=1).mean()
        out_cols["cross_embb_active_users"] = embb_df["active_users"]
        
    mmtc_df = other_slices.get("mMTC")
    if slice_type == "mMTC":
        mmtc_df = df
    if mmtc_df is not None:
        out_cols["cross_mmtc_active_devices"] = mmtc_df["active_users"]
        
    return pd.concat([df, pd.DataFrame(out_cols)], axis=1)

def build_features(df: pd.DataFrame, slice_type: str, other_slices: dict = None) -> pd.DataFrame:
    """Orchestrates all 8 feature categories. Returns feature matrix."""
    kpi_cols = ["dl_throughput", "latency", "jitter", "packet_loss", "prb_util", "active_users", "reliability"]
    kpi_cols = [c for c in kpi_cols if c in df.columns]
    
    df = add_lag_features(df, kpi_cols)
    df = add_rolling_stats(df, kpi_cols)
    df = add_ewma_features(df, kpi_cols)
    df = add_rate_of_change(df, kpi_cols)
    df = add_cyclical_time(df)
    
    sla_thresholds = SLICE_CONFIGS.get(slice_type).sla_thresholds if slice_type in SLICE_CONFIGS else []
    if sla_thresholds:
        df = add_sla_proximity(df, slice_type, sla_thresholds)
        
    df = add_cross_kpi_features(df, slice_type)
    df = add_cross_slice_features(df, slice_type, other_slices or {})
    
    return df

def process_all_slices(generator_kwargs: dict = None):
    """
    Reads from data/raw/generated/ and processes them, outputting to data/processed/.
    """
    raw_dir = Path("data/raw/generated")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    slices = {}
    for stype in ["eMBB", "URLLC", "mMTC"]:
        path = raw_dir / f"{stype.lower()}_synthetic.parquet"
        if path.exists():
            slices[stype] = pd.read_parquet(path)
            
    for stype, df in slices.items():
        others = {k: v for k, v in slices.items() if k != stype}
        feats = build_features(df, stype, others)
        feats.to_parquet(processed_dir / f"{stype}.parquet")

if __name__ == "__main__":
    process_all_slices()
