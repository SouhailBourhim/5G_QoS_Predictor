"""
Argus Flow Data Loader for 5G-NIDD Dataset

This module handles loading and preprocessing Argus network flow data
from the 5G-NIDD dataset and converting it to slice-level KPI time-series.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


class ArgusFlowLoader:
    """
    Loads and processes Argus network flow data from 5G-NIDD.
    """
    
    # Column mapping from Argus to our standardized names
    COLUMN_MAP = {
        # Temporal
        'StartTime': 'start_time',
        'LastTime': 'last_time',
        'Dur': 'duration_sec',
        'RunTime': 'run_time',
        'IdleTime': 'idle_time',
        
        # Traffic volume
        'TotBytes': 'total_bytes',
        'SrcBytes': 'src_bytes',
        'DstBytes': 'dst_bytes',
        'TotPkts': 'total_packets',
        'SrcPkts': 'src_packets',
        'DstPkts': 'dst_packets',
        
        # Application layer
        'TotAppByte': 'total_app_bytes',
        'SAppBytes': 'src_app_bytes',
        'DAppBytes': 'dst_app_bytes',
        
        # Rates
        'Rate': 'rate_bps',
        'SrcRate': 'src_rate_bps',
        'DstRate': 'dst_rate_bps',
        'Load': 'load',
        'SrcLoad': 'src_load',
        'DstLoad': 'dst_load',
        
        # Loss and retransmission
        'Loss': 'loss_packets',
        'SrcLoss': 'src_loss',
        'DstLoss': 'dst_loss',
        'pLoss': 'loss_pct',
        'Retrans': 'retrans_packets',
        'pRetran': 'retrans_pct',
        
        # Timing statistics
        'SIntPkt': 'src_interpkt_time',
        'DIntPkt': 'dst_interpkt_time',
        'SrcJitAct': 'src_jitter_active',
        'DstJitter': 'dst_jitter',
        'DstJitAct': 'dst_jitter_active',
        
        # TCP-specific
        'TcpRtt': 'tcp_rtt_ms',
        'SynAck': 'syn_ack_time',
        'AckDat': 'ack_dat_time',
        'State': 'tcp_state',
        'SrcWin': 'src_window',
        'DstWin': 'dst_window',
        
        # Packet sizes
        'sMeanPktSz': 'src_mean_pkt_size',
        'dMeanPktSz': 'dst_mean_pkt_size',
        
        # Identifiers
        'SrcAddr': 'src_ip',
        'DstAddr': 'dst_ip',
        'Sport': 'src_port',
        'Dport': 'dst_port',
        'Proto': 'protocol',
        
        # Labels
        'Label': 'label',
        'Attack Type': 'attack_type',
        'Attack Tool': 'attack_tool',
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the loader.
        
        Args:
            data_dir: Path to 5G-NIDD data directory
        """
        self.data_dir = data_dir or (RAW_DATA_DIR / "5g_nidd")
        
    def load_csv(
        self,
        file_path: Path,
        nrows: Optional[int] = None,
        sample_frac: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load a single Argus CSV file.
        
        Args:
            file_path: Path to CSV file
            nrows: Limit to first N rows
            sample_frac: Sample this fraction
            
        Returns:
            DataFrame with standardized column names
        """
        logger.info(f"Loading {file_path.name}")
        
        # Load CSV
        df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
        
        # Rename columns
        df = df.rename(columns=self.COLUMN_MAP)
        
        # Sample if requested
        if sample_frac and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=42)
        
        logger.info(f"Loaded: {len(df):,} flows")
        
        return df
    
    def filter_normal_traffic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for normal (non-attack) traffic only.
        
        Args:
            df: DataFrame with flow data
            
        Returns:
            Filtered DataFrame
        """
        if 'label' not in df.columns:
            logger.warning("No 'label' column found - returning all data")
            return df
        
        # Check unique labels
        unique_labels = df['label'].unique()
        logger.info(f"Found labels: {unique_labels}")
        
        # Filter for normal traffic
        normal_labels = ['normal', 'Normal', 'BENIGN', 'Benign', 'background', 'Background']
        df_normal = df[df['label'].isin(normal_labels)]
        
        removed = len(df) - len(df_normal)
        logger.info(f"Filtered to normal traffic: {len(df_normal):,} flows "
                   f"({removed:,} attack flows removed)")
        
        return df_normal
    
    def compute_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute QoS KPIs from flow-level features.
        
        Args:
            df: DataFrame with flow data
            
        Returns:
            DataFrame with added KPI columns
        """
        logger.info("Computing KPIs from flow data...")
        
        # Avoid division by zero
        duration_safe = df['duration_sec'].replace(0, np.nan)
        
        # ─── THROUGHPUT ───────────────────────────────────────────
        # Convert bytes to Mbps (megabits per second)
        df['dl_throughput_mbps'] = (df['dst_bytes'] / duration_safe) / 1e6 * 8
        df['ul_throughput_mbps'] = (df['src_bytes'] / duration_safe) / 1e6 * 8
        df['total_throughput_mbps'] = (df['total_bytes'] / duration_safe) / 1e6 * 8
        
        # ─── LATENCY ──────────────────────────────────────────────
        # Use TCP RTT if available
        if 'tcp_rtt_ms' in df.columns:
            df['latency_ms'] = df['tcp_rtt_ms']
        else:
            # Fallback: estimate from SYN-ACK timing
            if 'syn_ack_time' in df.columns:
                df['latency_ms'] = df['syn_ack_time']
            else:
                # Last resort: use inter-packet time as proxy
                df['latency_ms'] = df['src_interpkt_time'] * 1000  # Convert to ms
        
        # ─── JITTER ──────────────────────────────────────────────
        if 'dst_jitter' in df.columns:
            df['jitter_ms'] = df['dst_jitter']
        elif 'dst_jitter_active' in df.columns:
            df['jitter_ms'] = df['dst_jitter_active']
        else:
            df['jitter_ms'] = 0
        
        # ─── PACKET LOSS ──────────────────────────────────────────
        if 'loss_pct' in df.columns:
            df['packet_loss_pct'] = df['loss_pct']
        else:
            # Calculate from loss packets
            total_pkts_safe = df['total_packets'].replace(0, np.nan)
            df['packet_loss_pct'] = (df['loss_packets'] / total_pkts_safe) * 100
        
        # ─── PACKET RATE ──────────────────────────────────────────
        df['packet_rate'] = df['total_packets'] / duration_safe
        
        # ─── RELIABILITY (for URLLC) ──────────────────────────────
        # Inverse of loss rate
        df['reliability_pct'] = 100 - df['packet_loss_pct'].clip(0, 100)
        
        # Clean up infinities and extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Clip to reasonable ranges
        df['dl_throughput_mbps'] = df['dl_throughput_mbps'].clip(0, 10000)  # Max 10 Gbps
        df['ul_throughput_mbps'] = df['ul_throughput_mbps'].clip(0, 10000)
        df['latency_ms'] = df['latency_ms'].clip(0, 10000)  # Max 10 seconds
        df['jitter_ms'] = df['jitter_ms'].clip(0, 1000)
        df['packet_loss_pct'] = df['packet_loss_pct'].clip(0, 100)
        
        logger.info("KPI computation complete")
        
        return df
    
    def aggregate_to_timeseries(
        self,
        df: pd.DataFrame,
        window: str = '5min',
        timestamp_col: str = 'start_time'
    ) -> pd.DataFrame:
        """
        Aggregate flow-level data to time-series at specified granularity.
        
        This converts individual flows to slice-level KPIs over time windows.
        
        Args:
            df: DataFrame with flow data and computed KPIs
            window: Time window for aggregation (e.g., '5min', '1min')
            timestamp_col: Column containing timestamps
            
        Returns:
            Time-series DataFrame with aggregated KPIs
        """
        logger.info(f"Aggregating to {window} time-series...")
        
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            if pd.api.types.is_numeric_dtype(df[timestamp_col]):
                # Convert numeric Unix epoch timestamp to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
            else:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                except Exception:
                    # Handle duration offset format like "30:02.6" (mm:ss.ms)
                    def standardize_duration(x):
                        x_str = str(x)
                        parts = x_str.split(':')
                        if len(parts) == 2:
                            return '00:' + x_str
                        elif len(parts) == 1:
                            return '00:00:' + x_str
                        return x_str
                    
                    time_strs = df[timestamp_col].apply(standardize_duration)
                    deltas = pd.to_timedelta(time_strs)
                    # Use fixed base date so time series aggregation works properly
                    df[timestamp_col] = pd.Timestamp('2020-01-01') + deltas
        
        # Set timestamp as index
        df_indexed = df.set_index(timestamp_col)
        
        # Aggregation functions for each KPI
        def p95(x):
            return x.quantile(0.95)
            
        agg_funcs = {
            'dl_throughput_mbps': ['mean', 'median', 'std', 'max'],
            'ul_throughput_mbps': ['mean', 'median', 'std', 'max'],
            'latency_ms': ['mean', 'median', 'std', 'min', 'max', p95],
            'jitter_ms': ['mean', 'median', 'max'],
            'packet_loss_pct': ['mean', 'max'],
            'reliability_pct': ['mean', 'min'],
            'packet_rate': ['sum', 'mean'],
            'total_bytes': 'sum',
            'total_packets': 'sum',
        }
        
        # Count active flows (proxy for active users)
        agg_funcs['src_ip'] = 'nunique'  # Unique source IPs
        
        # Resample and aggregate
        df_agg = df_indexed.resample(window).agg(agg_funcs)
        
        # Flatten multi-level columns
        df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
        
        # Rename for clarity
        df_agg = df_agg.rename(columns={
            'src_ip_nunique': 'active_users',
        })
        
        # Reset index to make timestamp a column
        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={timestamp_col: 'timestamp'})
        
        # Fill missing windows with 0 (no traffic)
        df_agg = df_agg.fillna(0)
        
        logger.info(f"Aggregated to {len(df_agg)} time windows")
        
        return df_agg
    
    def load_and_process(
        self,
        file_path: Optional[Path] = None,
        nrows: Optional[int] = None,
        filter_normal: bool = True,
        aggregate_window: str = '5min'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete pipeline: load, filter, compute KPIs, aggregate.
        
        Args:
            file_path: CSV file to load. If None, auto-detect
            nrows: Limit rows
            filter_normal: Whether to filter for normal traffic only
            aggregate_window: Time window for aggregation
            
        Returns:
            Tuple of (flow_df, timeseries_df)
        """
        # Auto-detect file if not specified
        if file_path is None:
            csv_files = list(self.data_dir.rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
            # Use largest file
            file_path = max(csv_files, key=lambda p: p.stat().st_size)
            logger.info(f"Auto-selected: {file_path.name}")
        
        # Load
        df = self.load_csv(file_path, nrows=nrows)
        
        # Filter for normal traffic
        if filter_normal:
            df = self.filter_normal_traffic(df)
        
        # Compute KPIs
        df = self.compute_kpis(df)
        
        # Aggregate to time-series
        df_ts = self.aggregate_to_timeseries(df, window=aggregate_window)
        
        return df, df_ts


# ─── CONVENIENCE FUNCTIONS ────────────────────────────────────────────────────

def load_5g_nidd_flows(
    nrows: Optional[int] = None,
    filter_normal: bool = True
) -> pd.DataFrame:
    """Quick load of flow-level data."""
    loader = ArgusFlowLoader()
    df, _ = loader.load_and_process(nrows=nrows, filter_normal=filter_normal)
    return df


def load_5g_nidd_timeseries(
    window: str = '5min',
    nrows: Optional[int] = None,
    filter_normal: bool = True
) -> pd.DataFrame:
    """Quick load of aggregated time-series data."""
    loader = ArgusFlowLoader()
    _, df_ts = loader.load_and_process(
        nrows=nrows,
        filter_normal=filter_normal,
        aggregate_window=window
    )
    return df_ts


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("5G-NIDD ARGUS FLOW LOADER")
    print("="*70 + "\n")
    
    # Load and process
    loader = ArgusFlowLoader()
    df_flows, df_ts = loader.load_and_process(nrows=50000)
    
    print("\n📊 FLOW-LEVEL DATA")
    print("-" * 70)
    print(df_flows[['dl_throughput_mbps', 'ul_throughput_mbps', 'latency_ms', 
                    'jitter_ms', 'packet_loss_pct', 'reliability_pct']].describe())
    
    print("\n📈 TIME-SERIES DATA (5-min windows)")
    print("-" * 70)
    print(df_ts.head(10))
    
    # Save sample
    output_path = RAW_DATA_DIR / "5g_nidd" / "processed_timeseries_sample.csv"
    df_ts.to_csv(output_path, index=False)
    print(f"\n✅ Sample saved to: {output_path}")