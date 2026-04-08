# Task 1 — EDA & Calibration Parameter Extraction: Analysis

**Notebook:** `notebooks/01_eda_calibration.ipynb`
**Source data:** `data/raw/5g_nidd/` (5G-NIDD dataset, University of Oulu)
**Outputs:** `data/raw/5g_nidd/calibration_params.yaml`, `reports/figures/`

---

## What was done

The EDA pipeline loaded 5G-NIDD Argus flow records, computed per-KPI distribution statistics, fitted parametric distributions, computed a Pearson correlation matrix, estimated per-mobility-scenario variance, and saved all results to YAML and PNG.

---

## Key Findings

### 1. KPI Distributions are Highly Skewed

Every traffic KPI is dominated by near-zero values with rare but extreme spikes. This is consistent with real 5G network traffic: most flows are idle or low-rate, with occasional bursts.

| KPI | Distribution | Mean | Median | Max | Skewness |
|---|---|---|---|---|---|
| dl_throughput | lognormal | 2.41 Mbps | 0.016 Mbps | 1304 Mbps | 463 |
| ul_throughput | lognormal | 2.48 Mbps | 0.120 Mbps | 944 Mbps | — |
| latency | gamma | 0.00090 ms | 0.0 ms | 0.200 ms | 13.8 |
| packet_loss | beta | 0.35% | 0.0% | 62.5% | 8.6 |
| reliability | normal | 99.65% | 100.0% | 100.0% | — |
| active_users | normal | 13.0 | 14.0 | 30.0 | — |
| jitter | exponential | 0.0 | 0.0 | 0.0 | — |

The mean/median divergence for throughput is extreme: DL mean is 2.41 Mbps but median is only 0.016 Mbps, a 150× gap. This confirms the lognormal fit is appropriate — the distribution has a very long right tail driven by a small fraction of high-throughput flows.

### 2. Jitter is Effectively Zero in This Dataset

Jitter has mean = 0, std = 0, max = 0 across all records. The 5G-NIDD dataset does not capture jitter at the Argus flow level — it is a flow-record dataset, not a per-packet dataset. This is a known limitation. For the synthetic generator, jitter will need to be modeled independently using domain knowledge rather than calibrated from this data.

### 3. Latency Values are in Milliseconds but Very Small

The raw latency values (max 0.200 ms, mean 0.0009 ms) reflect Argus-computed RTT at the flow level, not end-to-end 5G radio latency. Real 5G radio latency is typically 1–30 ms for eMBB and sub-1 ms for URLLC. The calibration YAML captures these flow-level values, but the synthetic generator will need to scale them to realistic radio-layer ranges using the SLA thresholds defined in the design doc.

### 4. Packet Loss is Sparse but Has Heavy Tails

75th percentile of packet loss is 0% — meaning three-quarters of all flows have zero loss. But the 99th percentile jumps to 40% and the max is 62.5%. This bimodal behavior (mostly zero, occasionally very high) is well-captured by the beta distribution fit (a=0.098, b=21.6), which places most mass near zero with a long right tail.

### 5. Reliability is Near-Constant at 100%

Mean reliability is 99.65%, median is 100%, and the distribution is heavily left-skewed (most flows are perfectly reliable). The normal fit is technically correct but the distribution is not truly normal — it is a truncated distribution with a hard ceiling at 100%. The std of 3.92% is inflated by rare failure events (min = 37.5%). The synthetic generator should treat reliability as `100 - packet_loss` rather than sampling independently.

### 6. Throughput Range Spans 5 Orders of Magnitude

DL throughput ranges from 0 to 1304 Mbps, UL from 0 to 944 Mbps. The lognormal shape parameter `s` is very large (3.62 for DL, 2.01 for UL), indicating extreme spread. The 95th percentile is only 0.41 Mbps DL — meaning 95% of flows are below 0.41 Mbps. The top 1% starts at 8.15 Mbps. This is consistent with a dataset that mixes IoT-style low-rate flows with occasional video/bulk-transfer flows.

### 7. Correlation Structure

The correlation matrix was computed across flow-level KPIs. Key expected relationships:

- **Packet loss ↔ Reliability**: Strong negative correlation (−1.0 by construction, since reliability = 100 − packet_loss_pct in this dataset). This is a data artifact, not an independent signal.
- **DL ↔ UL throughput**: Moderate positive correlation expected (shared channel conditions), though the exact value depends on the traffic mix.
- **Latency ↔ Throughput**: Expected negative correlation (higher load → higher latency) but may be weak at flow level since most latency values are zero.

The near-zero median latency means the correlation signal between latency and throughput is likely weak in this dataset — the synthetic generator's physical KPI relationships (Layer 3) are more important than the empirical correlations here.

### 8. Mobility Variance: Proxy-Based, Not Ground Truth

The 5G-NIDD dataset has no explicit mobility column. The EDA module falls back to splitting rows into thirds as a proxy for vehicular / pedestrian / static scenarios. This means the mobility variance figures are **not** derived from actual mobility labels — they reflect temporal variation across the dataset's time range, not physical mobility differences.

Implication: the mobility variance output in `calibration_params.yaml` should be treated as a rough proxy. The synthetic generator's per-scenario variance should be set using domain knowledge (vehicular > pedestrian > static) rather than relying on these proxy values.

### 9. Active Users: Reasonable Range

Active users follows a near-normal distribution (mean=13, std=7.8, range 0–30). This is plausible for a base station dataset. The synthetic generator can use this directly for mMTC device counts, scaled appropriately per slice type.

---

## Implications for the Synthetic Generator (Task 2)

| Finding | Generator Impact |
|---|---|
| Jitter = 0 in dataset | Model jitter using domain knowledge: eMBB ~2–5 ms, URLLC ~0.1–0.5 ms |
| Latency values are flow-level RTT, not radio latency | Scale latency to SLA ranges: eMBB 5–30 ms, URLLC 0.5–5 ms, mMTC 50–1000 ms |
| Throughput is lognormal with extreme skew | Use lognormal noise but clip to physically plausible ranges per slice |
| Reliability = 100 − packet_loss | Do not sample reliability independently; derive from packet_loss |
| Packet loss is sparse (75th pct = 0%) | Use beta distribution for noise but ensure violation injection drives loss above SLA thresholds |
| Mobility variance is proxy-only | Set vehicular > pedestrian > static variance manually in generator config |

---

## Data Quality Notes

- **Row counts differ by KPI**: throughput columns have 9,944 rows (flow-level), while latency/packet_loss/reliability have 26,433 rows. This is because throughput is only defined for flows with actual data transfer, while latency and loss are computed for all flows including control traffic.
- **No jitter column**: The Argus flow format does not export per-flow jitter. The `jitter_ms` column is all zeros.
- **No explicit timestamps in flow records**: The time-series aggregation uses flow start times, which may not be uniformly spaced.

---

## Figures Generated

| File | Description |
|---|---|
| `reports/figures/timeseries_overview.png` | Aggregated KPI time-series across the full dataset window |
| `reports/figures/kpi_distributions.png` | Per-KPI histograms (all KPIs, grid layout) |
| `reports/figures/dist_dl_throughput_mbps.png` | DL throughput distribution detail |
| `reports/figures/dist_latency_ms.png` | Latency distribution detail |
| `reports/figures/dist_packet_loss_pct.png` | Packet loss distribution detail |

---

## Calibration YAML Status

`data/raw/5g_nidd/calibration_params.yaml` is populated and ready for consumption by the synthetic generator. All 7 KPI entries are present. The jitter entry has all-zero values and will need manual override in the generator.
