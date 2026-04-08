"""
Configuration management for 5G QoS Predictor project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


# ─── PROJECT PATHS ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ─── SLICE CONFIGURATIONS ─────────────────────────────────────────────────────

@dataclass
class SLAThreshold:
    """Single SLA threshold definition."""
    kpi_name: str
    threshold: float
    direction: str  # "min" (value must be >= threshold) or "max" (value must be <= threshold)
    window_minutes: int = 15  # SLA evaluated over this rolling window
    description: str = ""


@dataclass
class SliceConfig:
    """Configuration for a network slice type."""
    name: str
    description: str
    sla_thresholds: List[SLAThreshold]
    kpi_columns: List[str]
    daily_pattern: str
    weekend_factor: float
    base_prb_allocation: float
    max_prb_allocation: float


# ─── DEFAULT SLICE CONFIGURATIONS ─────────────────────────────────────────────

EMBB_CONFIG = SliceConfig(
    name="eMBB",
    description="Enhanced Mobile Broadband - video streaming, web browsing, downloads",
    sla_thresholds=[
        SLAThreshold("dl_throughput_mbps", 50.0, "min", 15, "Minimum 50 Mbps downlink"),
        SLAThreshold("latency_ms", 30.0, "max", 15, "Maximum 30 ms latency"),
        SLAThreshold("packet_loss_pct", 1.0, "max", 15, "Maximum 1% packet loss"),
    ],
    kpi_columns=[
        "dl_throughput_mbps", "ul_throughput_mbps", "latency_ms",
        "jitter_ms", "packet_loss_pct", "active_users", "prb_utilization_pct"
    ],
    daily_pattern="double_peak",
    weekend_factor=1.15,
    base_prb_allocation=50.0,
    max_prb_allocation=70.0,
)

URLLC_CONFIG = SliceConfig(
    name="URLLC",
    description="Ultra-Reliable Low-Latency Communications - factory automation, remote surgery",
    sla_thresholds=[
        SLAThreshold("latency_ms", 5.0, "max", 5, "Maximum 5 ms latency"),
        SLAThreshold("reliability_pct", 99.999, "min", 5, "Minimum 99.999% reliability"),
        SLAThreshold("jitter_ms", 1.0, "max", 5, "Maximum 1 ms jitter"),
    ],
    kpi_columns=[
        "latency_ms", "jitter_ms", "reliability_pct",
        "dl_throughput_mbps", "active_devices", "prb_utilization_pct"
    ],
    daily_pattern="business_hours",
    weekend_factor=0.3,
    base_prb_allocation=20.0,
    max_prb_allocation=30.0,
)

MMTC_CONFIG = SliceConfig(
    name="mMTC",
    description="Massive Machine-Type Communications - IoT sensors, smart meters",
    sla_thresholds=[
        SLAThreshold("delivery_rate_pct", 95.0, "min", 30, "Minimum 95% message delivery"),
        SLAThreshold("avg_latency_ms", 1000.0, "max", 30, "Maximum 1 second average latency"),
    ],
    kpi_columns=[
        "active_connections", "delivery_rate_pct", "avg_latency_ms", "prb_utilization_pct"
    ],
    daily_pattern="periodic_bursts",
    weekend_factor=0.95,
    base_prb_allocation=15.0,
    max_prb_allocation=25.0,
)

SLICE_CONFIGS = {
    "eMBB": EMBB_CONFIG,
    "URLLC": URLLC_CONFIG,
    "mMTC": MMTC_CONFIG,
}


# ─── DATA GENERATION CONFIG ───────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    start_date: str = "2024-01-01"
    duration_days: int = 90
    granularity_minutes: int = 5
    random_seed: int = 42
    
    # Violation event configuration
    events_per_week_range: tuple = (2, 6)
    buildup_minutes_range: tuple = (15, 90)
    duration_minutes_range: tuple = (15, 240)
    severity_range: tuple = (0.3, 1.0)


# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for model training."""
    prediction_horizons: List[int] = field(default_factory=lambda: [15, 30, 60])
    target_recall: float = 0.90
    
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 20,
    })
    
    # LightGBM parameters
    lgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })


# ─── FEATURE ENGINEERING CONFIG ───────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Lag steps (in number of time steps, not minutes)
    lag_steps: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 36, 72, 144])
    
    # Rolling window sizes (in number of time steps)
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 36, 72, 144, 288])
    
    # EWMA spans
    ewma_spans: List[int] = field(default_factory=lambda: [6, 12, 36])
    
    # Rate of change windows
    roc_windows: List[int] = field(default_factory=lambda: [1, 3, 6, 12])


# ─── EVALUATION CONFIG ────────────────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    train_days: int = 60
    val_days: int = 15
    test_days: int = 15
    
    # Expanding window CV
    cv_n_splits: int = 5
    cv_min_train_days: int = 14
    cv_test_days: int = 7


# ─── MAIN CONFIG CLASS ────────────────────────────────────────────────────────

@dataclass
class Config:
    """Main configuration container."""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            generation=GenerationConfig(**data.get("generation", {})),
            model=ModelConfig(**data.get("model", {})),
            features=FeatureConfig(**data.get("features", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = Config()