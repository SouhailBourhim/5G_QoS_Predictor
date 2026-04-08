# Task 2 — Synthetic Slice Data Generator: Analysis

**Notebook:** `notebooks/02_synthetic_generator.ipynb`
**Source data:** `data/raw/5g_nidd/calibration_params.yaml`
**Outputs:** `data/raw/generated/eMBB.parquet`, `data/raw/generated/URLLC.parquet`, `data/raw/generated/mMTC.parquet`

---

## What was done

The synthetic generator pipeline successfully modeled a realistic, continuous 5G network dataset spanning multiple slice types (eMBB, URLLC, mMTC). It incorporated physical KPI relationships, autoregressive noise based on empirically calibrated distributions from Task 1, and discrete event injections (e.g., flash crowds, equipment failures) to simulate SLA violations. Finally, the data was enriched with cross-slice coupling penalties when total network resources were saturated.

---

## Key Findings

### 1. Stable, High-Volume Data Generation
The generator produced exactly 25,920 timesteps per slice for a 90-day simulation. It enforces strict column types, including high-precision `datetime64[us]` for the timestamp and distinct labels for violation proximity (`any_breach`, `time_to_violation`, `violation_in_15min`, `violation_in_30min`, `violation_in_60min`).

### 2. Intraday Load Profiles Reflect Slice Characteristics
The base load dynamically adapts to the selected slice type:
- **eMBB**: Displays a triple-hump Gaussian curve mapping smoothly to morning commute, lunch, and evening streaming behaviors.
- **URLLC**: Features a sigmoid plateau reflecting business hours usage.
- **mMTC**: Modeled accurately with periodic impulse bursts common for machine-to-machine telemetry.

### 3. Realistic KPI Noise via Autoregression
Layer 4 AR(1) smoothing successfully maintains temporal coherence in KPI values rather than exhibiting pure white noise. This is vital since raw parameters extracted in Task 1 had high variance. Delay and loss values properly adhere to their respective baseline distributions without erratic non-sequential jumping.

### 4. Inject Events Simulate Accurate Degradations
The integration of 6 event types (like DDoS, fiber cuts, or software glitches) overlays linear degradation ramps onto KPIs. Depending on uniform severity scaling, this drives specific fields to breach predefined SLA limits dynamically.

### 5. Effective Cross-Slice PRB Coupling (Layer 6)
The visualization of stacked Physical Resource Block (PRB) utilization indicates successful competition modeling. When aggregate demand surpasses the 90% threshold, physical contention is appropriately penalized using a polynomial degradation factor, accurately cascading non-event-driven SLA violations across all slice profiles.

---

## Implications for the Feature Engineering Pipeline (Task 3)

| Finding | Feature Pipeline Impact |
|---|---|
| AR(1) Temporal Coherence is Present | Lag and rolling window features (e.g. EWMA spans) should provide strong predictive signals to capture the gradual trending. |
| Time-to-violation fields are Continuous | Can act as a proxy target for early validation and for engineering proximity-to-breach features. |
| PRB Bottlenecks are Systemic | Cross-slice aggregate PRB features must be engineered precisely to predict coupling degradations across slice boundaries. |
| Distinct Event Profiles | Rolling stats of rate of change features (diffs) will be crucial to capture the sudden but linear degradation ramps. |

---

## Visualizations Generated

| Plot Description | Insight Gained |
|---|---|
| Intraday Load Profiles per Slice | Confirms structural load shapes (triple-hump, plateau, impulses) correctly instantiate based on slice type. |
| 7-Day KPI Time-Series with Events | Demonstrates event injection degrades specific KPIs, correlating with violation labels visually over time. |
| Violation Label Distributions | Confirmed positive classification rates for `any_breach` and leading indicator labels are balanced and realistic. |
| Stacked PRB Utilization | Validates that aggregate total load crosses the 90% threshold dynamically, triggering cross-slice capacity penalties. |

---

## Conclusion
The data generation accurately mimics complex telecom constraints effectively. The generated parquet files represent a reliable, high-fidelity synthetic foundation matching SLA requirements tightly, acting as an optimal dataset for downstream feature extraction, temporal splitting, and predictive ML algorithm development.
