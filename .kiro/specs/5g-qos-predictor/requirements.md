# Requirements Document

## Introduction

The 5G QoS Predictor is an end-to-end machine learning system that predicts SLA (Service Level Agreement) violations in 5G network slices before they occur, providing 15 to 60 minutes of advance warning. It is a prototype of a 3GPP NWDAF (Network Data Analytics Function) analytics module implementing the QoS Sustainability analytics ID defined in 3GPP TS 23.288 §6.9.

The system covers: synthetic data generation calibrated from real 5G-NIDD measurements, domain-informed feature engineering (~254 features across 8 categories), multi-horizon binary classification (15 min, 30 min, 60 min ahead), a FastAPI REST API, a Streamlit monitoring dashboard, and Docker containerization.

---

## Glossary

- **System**: The 5G QoS Predictor end-to-end ML system.
- **NWDAF**: Network Data Analytics Function — a 3GPP-defined logical function that collects and analyzes network data to support network management.
- **SLA**: Service Level Agreement — a contractual commitment on network KPI thresholds for a given slice.
- **SLA Violation**: A timestep at which one or more KPIs breach their SLA threshold for the slice type.
- **Slice / Network Slice**: A virtualized end-to-end network partition with dedicated QoS guarantees. Three types exist: eMBB, URLLC, and mMTC.
- **eMBB**: Enhanced Mobile Broadband slice — optimized for high-throughput use cases (video streaming, web browsing). SLA: DL throughput ≥ 50 Mbps, latency ≤ 30 ms, packet loss ≤ 1%.
- **URLLC**: Ultra-Reliable Low-Latency Communications slice — optimized for mission-critical use cases (factory robots, remote surgery). SLA: latency ≤ 5 ms, reliability ≥ 99.999%, jitter ≤ 1 ms.
- **mMTC**: Massive Machine-Type Communications slice — optimized for IoT density (smart meters, sensors). SLA: message delivery ≥ 95%, latency ≤ 1000 ms.
- **KPI**: Key Performance Indicator — a measurable network metric (e.g., DL throughput, latency, jitter, packet loss, PRB utilization, reliability, active users/devices).
- **PRB**: Physical Resource Block — the basic unit of radio resource allocation in 5G NR.
- **5G-NIDD**: A publicly available 5G network intrusion detection dataset from the University of Oulu, used as the calibration source for the synthetic generator.
- **Calibration_Params**: Statistical parameters (distribution type, mean, std, min, max) extracted from 5G-NIDD and stored in `data/raw/5g_nidd/calibration_params.yaml`, used to configure the Synthetic_Generator.
- **Synthetic_Generator**: The module (`src/data/generator.py`) that produces synthetic 5-minute-granularity KPI time-series for all three slice types.
- **Feature_Pipeline**: The module (`src/features/engineering.py`) that transforms raw KPI time-series into the full ~254-feature matrix with no lookahead.
- **Classifier**: The XGBoost binary classification model (`src/models/classifier.py`) that predicts SLA violation probability for a given slice and prediction horizon.
- **Forecaster**: The XGBoost regression model (`src/models/forecaster.py`) that predicts future KPI values for a given slice, KPI, and horizon.
- **LSTM_Classifier**: The optional PyTorch LSTM model (`src/models/lstm.py`) that classifies SLA violations using a 2-hour sequence window.
- **Evaluator**: The evaluation framework (`src/evaluation/evaluate.py`) that computes all 8 evaluation pillars.
- **API**: The FastAPI REST service (`src/deployment/api.py`) that exposes prediction endpoints.
- **Dashboard**: The Streamlit web application (`src/deployment/dashboard.py`) providing a 6-page monitoring interface.
- **Temporal_Split**: The strict chronological partitioning of data into train (days 1–60), validation (days 61–75), and test (days 76–90) sets with no overlap.
- **Horizon**: A prediction lookahead window. Three horizons are used: 15 min, 30 min, and 60 min.
- **Event_Type**: A labeled category of network disturbance injected during synthetic data generation. Six types: traffic_surge, gradual_congestion, interference, hw_degradation, resource_starvation, iot_storm. Normal operation is labeled "normal".
- **AUC-PR**: Area Under the Precision-Recall Curve — the primary model evaluation metric, preferred over AUC-ROC due to class imbalance.
- **SHAP**: SHapley Additive exPlanations — a framework for explaining individual model predictions. Must use `TreeExplainer` for XGBoost models.
- **Threshold_Optimizer**: The procedure that selects the classification decision threshold on the validation set to maximize precision subject to recall ≥ 90%.
- **Health_Status**: A three-level risk label derived from the 30-minute violation probability: "healthy" (< 30%), "warning" (30–70%), "critical" (> 70%).

---

## Requirements

### Requirement 1: EDA and Calibration Parameter Extraction

**User Story:** As a data scientist, I want to extract statistical calibration parameters from the real 5G-NIDD dataset, so that the synthetic data generator produces realistic KPI distributions.

#### Acceptance Criteria

1. THE EDA_Module SHALL compute per-KPI distribution parameters (distribution type, mean, std, min, max, range) from the 5G-NIDD dataset for each of the following KPIs: DL throughput, RTT/latency, jitter, packet loss, PRB utilization, active users, and reliability.
2. THE EDA_Module SHALL compute inter-KPI Pearson correlation coefficients, including the positive correlation between RSRP and DL throughput, and the positive correlation between load and latency.
3. THE EDA_Module SHALL compute per-mobility-scenario (vehicular, pedestrian, static) variance for each KPI and verify that vehicular variance exceeds static variance.
4. WHEN EDA completes, THE EDA_Module SHALL persist all calibration parameters to `data/raw/5g_nidd/calibration_params.yaml` in a structured format importable by the Synthetic_Generator.
5. THE EDA_Module SHALL produce at minimum one time-series overview plot and one per-KPI distribution plot, saved to `reports/figures/`.

---

### Requirement 2: Synthetic Slice Data Generation

**User Story:** As a data scientist, I want to generate 90 days of synthetic 5-minute-granularity KPI data for all three slice types, so that I have a labeled training dataset with known ground truth for SLA violations.

#### Acceptance Criteria

1. THE Synthetic_Generator SHALL produce exactly 25,920 rows per slice (90 days × 288 five-minute intervals per day) for each of the three slice types: eMBB, URLLC, and mMTC.
2. THE Synthetic_Generator SHALL model intraday load profiles using slice-specific patterns: a triple-hump Gaussian profile for eMBB, a sigmoid business-hours plateau for URLLC, and periodic impulse bursts for mMTC.
3. THE Synthetic_Generator SHALL apply weekly modulation factors per slice type: eMBB weekday 1.00 / weekend 1.15, URLLC weekday 1.00 / weekend 0.30, mMTC weekday 1.00 / weekend 0.95, with an additional per-day random variation drawn from Normal(1.0, 0.05).
4. THE Synthetic_Generator SHALL derive KPI values from the load level using physical relationships: KPIs that increase with load (latency, jitter, packet_loss, prb_util, active_users) use V_base = V_min + (V_max − V_min) × load × weekly_factor; KPIs that decrease with load (dl_throughput, reliability) use V_base = V_max − (V_max − V_min) × load × weekly_factor.
5. THE Synthetic_Generator SHALL apply temporally autocorrelated noise with smoothing coefficient α = 0.3, where each noise sample is drawn from the KPI-specific distribution specified in Calibration_Params.
6. THE Synthetic_Generator SHALL inject six event types — traffic_surge, gradual_congestion, interference, hw_degradation, resource_starvation, iot_storm — with the onset timing, buildup duration, and affected slices defined in the project specification.
7. THE Synthetic_Generator SHALL compute degradation during event buildup using a linear ramp: degradation = severity × 0.7 × progress during buildup, and severity × Normal(1.0, 0.1) during the active event window, where severity is drawn from Uniform(0.3, 1.0) per event.
8. THE Synthetic_Generator SHALL tag every generated row with its event_type label (one of the six event types or "normal").
9. THE Synthetic_Generator SHALL implement cross-slice PRB coupling: when total PRB utilization across all three slices exceeds 90%, the excess is applied as a penalty to eMBB throughput and latency with a multiplier of 1.5.
10. THE Synthetic_Generator SHALL construct three binary horizon target columns per row: `violation_in_15min`, `violation_in_30min`, `violation_in_60min`, where a label of 1 indicates that a SLA violation will occur within the respective horizon.
11. THE Synthetic_Generator SHALL construct a `time_to_violation` column representing the number of minutes until the next SLA violation, or a sentinel value when no violation is imminent.
12. THE Synthetic_Generator SHALL store all output datasets in Parquet format under `data/raw/generated/`.
13. WHEN the Synthetic_Generator runs with violation injection enabled, THE Synthetic_Generator SHALL produce a `any_breach` positive rate between 2% and 10% inclusive for each slice.
14. THE Synthetic_Generator SHALL load all KPI distribution parameters from Calibration_Params and SHALL NOT hardcode distribution parameters in the generator source code.

---

### Requirement 3: Feature Engineering Pipeline

**User Story:** As a data scientist, I want to transform raw KPI time-series into a rich feature matrix, so that the classifier has sufficient domain-informed signal to predict SLA violations.

#### Acceptance Criteria

1. THE Feature_Pipeline SHALL compute temporal lag features for lag steps [1, 3, 6, 12, 24, 36, 72, 144] for each KPI column, producing 56 lag features total (7 KPIs × 8 lags).
2. THE Feature_Pipeline SHALL compute rolling statistics (mean, std, range, coefficient of variation) for window sizes [6, 12, 36, 72, 144, 288] for each KPI column, producing 84 rolling features total (7 KPIs × 6 windows × 4 statistics).
3. THE Feature_Pipeline SHALL compute EWMA features for spans [6, 12, 36] for each KPI column, producing 21 EWMA features total (7 KPIs × 3 spans).
4. THE Feature_Pipeline SHALL compute rate-of-change features for each KPI: first difference (diff1), 30-minute difference (diff6), second difference (acceleration diff2), and a smoothed trend direction sign over a 6-step rolling window, producing 35 rate-of-change features total (7 KPIs × 5 features).
5. THE Feature_Pipeline SHALL compute 8 cyclical time encoding features: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_business_hours, is_peak_evening, is_off_peak.
6. THE Feature_Pipeline SHALL compute SLA proximity features for each KPI–threshold pair: sla_margin (signed distance to threshold), sla_margin_norm (normalized margin), rolling minimum margin for windows [6, 12, 36], and time_to_breach (estimated minutes until SLA crossing at current rate, clipped to [0, 999]), producing at minimum 30 SLA proximity features.
7. THE Feature_Pipeline SHALL compute cross-KPI features for eMBB: bandwidth-delay product (BDP), spectral efficiency, effective throughput, jitter ratio, and BDP 30-minute difference, producing 5 cross-KPI features for eMBB.
8. THE Feature_Pipeline SHALL compute cross-slice features when predicting for a given slice: competitor PRB utilization from other slices, total system PRB utilization, rolling mean PRB utilization for windows [6, 12, 36] from the eMBB slice, eMBB active users, and mMTC active devices, producing 12 cross-slice features.
9. THE Feature_Pipeline SHALL use only `shift()` and `rolling()` operations with `min_periods=1` and SHALL NOT use any `iloc` indexing that accesses future timesteps.
10. WHEN the Feature_Pipeline processes a dataset, THE Feature_Pipeline SHALL produce a feature matrix with at least 200 columns.
11. THE Feature_Pipeline SHALL store all engineered feature datasets in Parquet format under `data/processed/`.
12. FOR ALL lag features produced by the Feature_Pipeline, the lag step SHALL be greater than or equal to 1, ensuring no zero-lag or negative-lag features exist.

---

### Requirement 4: Temporal Data Splitting

**User Story:** As a data scientist, I want to split the dataset using strict temporal partitioning, so that no future information leaks into the training or validation sets.

#### Acceptance Criteria

1. THE Temporal_Split SHALL partition the 90-day dataset into three non-overlapping, chronologically ordered sets: train (days 1–60), validation (days 61–75), and test (days 76–90).
2. THE Temporal_Split SHALL sort the dataset by timestamp before partitioning.
3. WHEN the Temporal_Split is applied, THE Temporal_Split SHALL assert that the maximum timestamp in the train set is strictly less than the minimum timestamp in the validation set.
4. WHEN the Temporal_Split is applied, THE Temporal_Split SHALL assert that the maximum timestamp in the validation set is strictly less than the minimum timestamp in the test set.
5. THE Temporal_Split SHALL store the resulting train, validation, and test Parquet files under `data/splits/`.
6. THE Evaluator SHALL verify that the `violation_in_30min` positive rate in each split is between 1% and 15% inclusive before reporting any evaluation metric.

---

### Requirement 5: KPI Forecaster (Regression)

**User Story:** As a data scientist, I want to train XGBoost regression models that predict future KPI values, so that the system can provide forecasted KPI trajectories alongside violation probabilities.

#### Acceptance Criteria

1. THE Forecaster SHALL train one XGBoost regression model per (slice_type, KPI, horizon) combination, where horizons are [3, 6, 12] steps corresponding to [15 min, 30 min, 60 min].
2. THE Forecaster SHALL construct the regression target for each (KPI, horizon) pair by shifting the KPI column by −h steps (future value at time t+h).
3. THE Forecaster SHALL train using only the train split and evaluate on the validation split for early stopping with `early_stopping_rounds=20` and `eval_metric='rmse'`.
4. THE Forecaster SHALL report MAE, RMSE, and MAPE per (slice_type, KPI, horizon) combination on the test split.
5. THE Forecaster SHALL save all trained model artifacts to `models/`.

---

### Requirement 6: SLA Violation Classifier (Binary Classification)

**User Story:** As a data scientist, I want to train XGBoost classifiers that predict SLA violation probability at multiple horizons, so that operators receive advance warning with sufficient lead time.

#### Acceptance Criteria

1. THE Classifier SHALL train one XGBoost binary classification model per (slice_type, horizon) combination, where horizons are [15 min, 30 min, 60 min].
2. THE Classifier SHALL compute `scale_pos_weight` as the ratio of negative to positive samples in the train split to handle the ~3–5% positive class rate.
3. THE Classifier SHALL use `eval_metric='aucpr'` and `early_stopping_rounds=20`, training on the train split and using the validation split for early stopping.
4. THE Threshold_Optimizer SHALL select the decision threshold on the validation split as the highest-precision threshold that achieves recall ≥ 90%, and SHALL NOT use the test split for threshold selection.
5. WHEN no threshold achieves recall ≥ 90% on the validation split, THE Threshold_Optimizer SHALL fall back to a threshold of 0.5.
6. THE Classifier SHALL save all trained model artifacts and their corresponding optimal thresholds to `models/`.
7. THE Classifier SHALL use AUC-PR as the primary reported metric, not AUC-ROC.

---

### Requirement 7: Optional LSTM Classifier

**User Story:** As a data scientist, I want an optional LSTM-based sequence classifier, so that I can compare sequence-aware predictions against the XGBoost baseline.

#### Acceptance Criteria

1. WHERE the LSTM option is enabled, THE LSTM_Classifier SHALL use the last 24 timesteps (2 hours) as the input sequence window.
2. WHERE the LSTM option is enabled, THE LSTM_Classifier SHALL implement a two-layer LSTM with hidden size 128, followed by fully connected layers (128 → 64 → 32 → 1) with ReLU activations and a sigmoid output.
3. WHERE the LSTM option is enabled, THE LSTM_Classifier SHALL apply dropout with rate 0.2 between LSTM layers.
4. WHERE the LSTM option is enabled, THE LSTM_Classifier SHALL train using the same Temporal_Split as the XGBoost Classifier and SHALL NOT use random shuffling of the time-series sequence.

---

### Requirement 8: Evaluation Framework

**User Story:** As a data scientist, I want a comprehensive 8-pillar evaluation framework, so that model performance is reported rigorously and comparably across slices, horizons, and event types.

#### Acceptance Criteria

1. THE Evaluator SHALL verify temporal integrity (Pillar 1) before computing any other metric, asserting no timestamp overlap between splits and that violation rates are within expected bounds.
2. THE Evaluator SHALL compute overall classification metrics (Pillar 2): precision, recall, F1, AUC-ROC, AUC-PR, and confusion matrix on the test split using the optimal threshold from the Threshold_Optimizer.
3. THE Evaluator SHALL report all Pillar 2 metrics separately for each slice type (eMBB, URLLC, mMTC) as Pillar 3 (per-slice analysis).
4. THE Evaluator SHALL compute per-event-type recall (Pillar 4) for each of the seven event_type labels (traffic_surge, gradual_congestion, interference, hw_degradation, resource_starvation, iot_storm, normal) using the event_type column from the test split.
5. THE Evaluator SHALL report F1 scores at all three horizons (15 min, 30 min, 60 min) as Pillar 5 (horizon analysis).
6. THE Evaluator SHALL compute early warning lead time statistics (Pillar 6) for all true positive predictions: median lead time, mean lead time, and 25th/75th percentile lead times in minutes.
7. THE Evaluator SHALL compute SHAP feature importance (Pillar 7) using `shap.TreeExplainer` for XGBoost models and SHALL NOT use the generic `shap.Explainer`.
8. THE Evaluator SHALL compare XGBoost performance against a static threshold baseline (Pillar 8), where the baseline triggers an alert when any KPI is within 15% of its SLA threshold.
9. THE Evaluator SHALL produce a global SHAP summary bar plot saved to `reports/figures/`.

---

### Requirement 9: FastAPI REST API

**User Story:** As a NOC operator or orchestrator, I want a REST API that accepts recent KPI history and returns violation predictions, so that the system can be integrated into automated network management workflows.

#### Acceptance Criteria

1. THE API SHALL expose a POST `/predict` endpoint that accepts a `KPIHistory` payload containing: `slice_type` (string), `timestamps` (list of ISO-format strings), and `kpi_values` (dict mapping KPI name to list of float values).
2. WHEN a `/predict` request is received with fewer than 12 timesteps, THE API SHALL return an HTTP 422 error with a descriptive validation message.
3. WHEN a valid `/predict` request is received, THE API SHALL return a `PredictionResponse` containing: `slice_type`, `violation_probability` (dict with keys "15min", "30min", "60min"), `forecasted_kpis` (dict of KPI to horizon-keyed predicted values), `top_risk_factors` (list of top-5 SHAP feature–value pairs), `recommendations` (list of actionable strings), and `health_status` (one of "healthy", "warning", "critical").
4. THE API SHALL derive `health_status` from the 30-minute violation probability: "healthy" when probability < 0.30, "warning" when probability is between 0.30 and 0.70 inclusive, and "critical" when probability > 0.70.
5. THE API SHALL expose a GET `/health/{slice_type}` endpoint that returns the current health status for the specified slice type.
6. THE API SHALL expose a GET `/slices` endpoint that returns the list of supported slice types.
7. THE API SHALL generate recommendations by mapping top SHAP features to the predefined recommendation strings defined in the project specification.
8. WHEN an unsupported `slice_type` is provided to any endpoint, THE API SHALL return an HTTP 404 error.

---

### Requirement 10: Streamlit Dashboard

**User Story:** As a NOC analyst, I want a multi-page web dashboard, so that I can monitor slice health, visualize KPI trends, and understand model predictions in real time.

#### Acceptance Criteria

1. THE Dashboard SHALL implement six pages: Slice Overview, Real-time Monitoring, Violation Prediction, Model Performance, Batch Analysis, and Feature Importance.
2. THE Dashboard SHALL display a risk gauge on the Slice Overview page for each slice type, color-coded green for probability < 30%, yellow for 30–70%, and red for > 70%.
3. THE Dashboard SHALL display KPI time-series plots with SLA threshold lines overlaid on the Real-time Monitoring page.
4. THE Dashboard SHALL display a side-by-side comparison of all three slices on the Slice Overview page.
5. THE Dashboard SHALL display the violation probability timeline alongside actual KPI values on the Violation Prediction page.
6. THE Dashboard SHALL display confusion matrices, per-event-type recall, and lead time distribution on the Model Performance page.
7. THE Dashboard SHALL accept CSV file uploads for historical backtesting on the Batch Analysis page.
8. THE Dashboard SHALL display SHAP summary plots and individual prediction explanations on the Feature Importance page.

---

### Requirement 11: Docker Containerization

**User Story:** As a DevOps engineer, I want the API and dashboard containerized with Docker Compose, so that the full system can be deployed reproducibly with a single command.

#### Acceptance Criteria

1. THE System SHALL provide a `Dockerfile.api` that builds the FastAPI service and exposes port 8000.
2. THE System SHALL provide a `Dockerfile.dashboard` that builds the Streamlit dashboard and exposes port 8501.
3. THE System SHALL provide a `docker-compose.yml` that defines both services, mounts the `models/` directory into the API container, and configures the Dashboard to connect to the API via the `API_URL` environment variable.
4. WHEN the API container starts, THE API SHALL pass a health check by responding to a GET request to `/slices`.
5. THE Dashboard service SHALL declare a dependency on the API service in `docker-compose.yml`.
6. WHEN `docker-compose up --build` is executed, THE System SHALL make the API documentation available at `http://localhost:8000/docs` and the Dashboard at `http://localhost:8501`.

---

### Requirement 12: Data Integrity and Anti-Leakage Constraints

**User Story:** As a data scientist, I want strict data integrity guarantees enforced throughout the pipeline, so that reported evaluation metrics reflect true generalization performance.

#### Acceptance Criteria

1. THE System SHALL use strict temporal partitioning for all train/validation/test splits and SHALL NOT use random shuffling or random sampling for split construction.
2. THE Threshold_Optimizer SHALL select the decision threshold exclusively on the validation split and SHALL NOT access the test split during threshold selection.
3. THE Feature_Pipeline SHALL compute all features using only data at time t and earlier, with no access to data at time t+1 or later.
4. THE Synthetic_Generator SHALL store all datasets in Parquet format and SHALL NOT use CSV format for any dataset larger than a diagnostic sample.
5. THE Evaluator SHALL use AUC-PR as the primary reported metric for all classifier comparisons due to the ~3–5% positive class rate.
6. THE Evaluator SHALL use `shap.TreeExplainer` for all SHAP computations on XGBoost and LightGBM models.
7. THE Synthetic_Generator SHALL tag every generated row with an `event_type` column, and this column SHALL be present in all downstream splits and evaluation datasets.

---

### Requirement 13: Testing

**User Story:** As a developer, I want automated tests covering temporal integrity, feature correctness, and data generation validity, so that regressions in pipeline correctness are caught early.

#### Acceptance Criteria

1. THE Test_Suite SHALL include a test that verifies no timestamp overlap exists between the train, validation, and test splits produced by the Temporal_Split.
2. THE Test_Suite SHALL include a test that verifies the `any_breach` positive rate in a 90-day eMBB synthetic dataset with violation injection enabled is between 2% and 10% inclusive.
3. THE Test_Suite SHALL include a test that verifies all lag feature column names produced by the Feature_Pipeline have a lag step of at least 1.
4. THE Test_Suite SHALL include a test that verifies the SLA margin sign convention: a KPI value safely above a minimum threshold produces a positive `sla_margin` value.
5. THE Test_Suite SHALL include a test that verifies the Feature_Pipeline produces at least 200 feature columns for a valid input dataset.
6. THE Test_Suite SHALL include a test that verifies a 30-day synthetic eMBB dataset generated without violation injection has DL throughput values of at least 45 Mbps for all rows.
