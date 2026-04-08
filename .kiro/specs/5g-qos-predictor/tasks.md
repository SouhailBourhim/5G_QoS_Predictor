# Implementation Plan: 5G QoS Predictor

## Overview

Implement the end-to-end ML pipeline in sequential phases: EDA/calibration → synthetic data generation → feature engineering → temporal splitting → XGBoost classifier + forecaster → evaluation → FastAPI → Streamlit dashboard → Docker. Each phase builds on the previous, with tests placed close to the code they validate.

## Tasks

- [x] 1. EDA module and calibration parameter extraction
  - Implement `src/data/eda_nidd.py` with functions: `run_eda`, `compute_kpi_stats`, `compute_correlations`, `compute_mobility_variance`, `save_calibration_params`, `save_plots`
  - Load 5G-NIDD Argus flow records via the existing `ArgusFlowLoader` in `src/data/argus_loader.py`
  - Fit per-KPI distributions (log-normal for throughput, gamma for latency) and compute mean, std, min, max, range
  - Compute Pearson correlation matrix and per-mobility-scenario variance
  - Write output to `data/raw/5g_nidd/calibration_params.yaml` and plots to `reports/figures/`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.N Notebook — EDA and calibration visualization (`notebooks/01_eda_calibration.ipynb`)
  - Call `run_eda()` and display the returned calibration params as a formatted table
  - Plot per-KPI histograms with the fitted distribution curve overlaid
  - Display the Pearson correlation heatmap
  - Show per-mobility-scenario variance bar charts
  - Render the time-series overview plot inline

- [x] 2. Synthetic slice data generator
  - [x] 2.1 Implement `src/data/generator.py` with all six layers
    - Implement `build_load_profile` (Layer 1+2): intraday patterns for eMBB (triple-hump Gaussian), URLLC (sigmoid plateau), mMTC (periodic impulse), weekly modulation factors, and per-day Normal(1.0, 0.05) jitter
    - Implement `derive_kpis_from_load` (Layer 3): physical KPI relationships (proportional vs. inverse with load)
    - Implement `apply_autocorrelated_noise` (Layer 4): AR(1) smoothing with α=0.3 using calibration_params.yaml distributions
    - Implement `inject_events` (Layer 5): all six event types with linear degradation ramps and Uniform(0.3, 1.0) severity
    - Implement `apply_cross_slice_coupling` (Layer 6): PRB penalty when total PRB > 90%
    - Implement `build_targets`: `any_breach`, `time_to_violation`, `violation_in_{15,30,60}min` columns
    - Implement `generate_all_slices`: orchestrate all layers, apply cross-slice coupling, save Parquet to `data/raw/generated/`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.14_

  - [x] 2.2 Write unit tests for the synthetic generator
    - Test that `generate_slice_data('eMBB', days=90)` produces exactly 25,920 rows — _Requirements: 2.1_
    - Test that `any_breach` positive rate is between 2% and 10% for a 90-day eMBB run with injection enabled — _Requirements: 2.13, 13.2_
    - Test that a 30-day eMBB run without injection has `dl_throughput >= 45` for all rows — _Requirements: 13.6_
    - Test that every row has a non-null `event_type` column — _Requirements: 2.8, 12.7_

- [x] 2.N Notebook — Synthetic data visualization (`notebooks/02_synthetic_generator.ipynb`)
  - Call `generate_all_slices()` and display row counts and schema for each slice
  - Plot the intraday load profile for each slice type (eMBB triple-hump, URLLC plateau, mMTC impulses)
  - Plot KPI time-series for a 7-day window per slice with event annotations
  - Show `any_breach` positive rate and `violation_in_{15,30,60}min` label distributions as bar charts
  - Visualize cross-slice PRB coupling: stacked PRB area chart with the 90% threshold line

- [ ] 3. Feature engineering pipeline
  - [ ] 3.1 Implement `src/features/engineering.py` with all 8 feature categories
    - Implement `add_lag_features`: lags [1,3,6,12,24,36,72,144] for each KPI → 56 features
    - Implement `add_rolling_stats`: windows [6,12,36,72,144,288] × {mean, std, range, cv} → 84 features
    - Implement `add_ewma_features`: spans [6,12,36] → 21 features
    - Implement `add_rate_of_change`: diff1, diff6, diff2, trend_sign → 35 features
    - Implement `add_cyclical_time`: hour_sin/cos, dow_sin/cos, is_weekend, is_business_hours, is_peak_evening, is_off_peak → 8 features
    - Implement `add_sla_proximity`: sla_margin, sla_margin_norm, rolling min margins [6,12,36], time_to_breach → ≥30 features
    - Implement `add_cross_kpi_features` (eMBB only): BDP, spectral_eff, eff_throughput, jitter_ratio, bdp_diff → 5 features
    - Implement `add_cross_slice_features`: competitor PRB, total PRB, rolling PRB means, active users/devices → 12 features
    - Implement `build_features` orchestrator; save output Parquet to `data/processed/`
    - All operations must use only `shift()` and `rolling(min_periods=1)` — no future-index `iloc`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11_

  - [ ]* 3.2 Write unit tests for the feature pipeline
    - Test that all lag feature column names have lag step ≥ 1 — _Requirements: 3.12, 13.3_
    - Test that `sla_margin` is positive for a KPI safely above its minimum threshold — _Requirements: 13.4_
    - Test that `build_features` produces at least 200 columns for a valid input — _Requirements: 3.10, 13.5_

- [ ]* 3.N Notebook — Feature engineering visualization (`notebooks/03_feature_engineering.ipynb`)
  - Call `build_features()` on the eMBB slice and display the feature matrix shape and column list
  - Plot a sample of lag features vs. the raw KPI to confirm correct shift direction
  - Show rolling-mean and EWMA smoothing overlaid on the raw latency series
  - Display SLA proximity features: `sla_margin` and `time_to_breach` over a 24-hour window
  - Render a feature correlation heatmap (top-50 features by variance)

- [ ] 4. Temporal data splitting
  - Implement `temporal_split` in `src/data/generator.py` (or a dedicated `src/data/splitter.py`)
  - Sort by timestamp, cut at day 60 (train) and day 75 (val), remainder is test
  - Assert `train.timestamp.max() < val.timestamp.min()` and `val.timestamp.max() < test.timestamp.min()`
  - Save `{slice_type}_{train|val|test}.parquet` to `data/splits/`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 4.1 Write unit test for temporal split integrity
    - Test that no timestamp overlap exists between any two splits — _Requirements: 4.3, 4.4, 13.1_

- [ ]* 4.N Notebook — Temporal split visualization (`notebooks/04_temporal_split.ipynb`)
  - Call `temporal_split()` and print the timestamp ranges and row counts for each split
  - Plot a timeline bar showing train / val / test boundaries for each slice
  - Show `violation_in_30min` positive rate per split as a bar chart to confirm no leakage drift
  - Display a KPI time-series with vertical lines marking the split boundaries

- [ ] 5. Checkpoint — verify data pipeline end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. XGBoost SLA violation classifier
  - [ ] 6.1 Implement `src/models/classifier.py`
    - Implement `train_classifier`: XGBoost with `scale_pos_weight`, `eval_metric='aucpr'`, `early_stopping_rounds=20`; hyperparameters as specified in design
    - Implement `find_optimal_threshold`: highest-precision threshold achieving recall ≥ 0.90 on val set; fallback to 0.5
    - Implement `save_classifier` / `load_classifier`: save model as `{slice_type}_clf_{horizon}min.json` + `.threshold` file
    - Train 9 models total (3 slices × 3 horizons: 15min, 30min, 60min)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 12.2_

  - [ ]* 6.2 Write unit tests for the classifier
    - Test that `find_optimal_threshold` returns a value in [0, 1]
    - Test that `find_optimal_threshold` falls back to 0.5 when no threshold meets the recall constraint
    - _Requirements: 6.4, 6.5_

- [ ]* 6.N Notebook — Classifier training and evaluation (`notebooks/05_classifier.ipynb`)
  - Train the eMBB 30-min classifier and plot the AUC-PR and AUC-ROC curves
  - Plot the precision-recall trade-off curve with the selected threshold marked
  - Display the confusion matrix for each slice × horizon combination
  - Show feature importance (top-20 XGBoost gain scores) as a horizontal bar chart

- [ ] 7. XGBoost KPI forecaster
  - Implement `src/models/forecaster.py`
  - Implement `train_forecaster`: XGBoost regressor, target = `df[kpi].shift(-h)`, early stopping on RMSE
  - Implement `evaluate_forecaster`: returns `{mae, rmse, mape}` on test split
  - Train up to 63 models (3 slices × 7 KPIs × 3 horizons); save artifacts to `models/`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 7.N Notebook — Forecaster evaluation (`notebooks/06_forecaster.ipynb`)
  - Plot actual vs. predicted KPI values for eMBB dl_throughput at all three horizons
  - Display a MAE / RMSE / MAPE summary table across all slice × KPI × horizon combinations
  - Show residual distribution histograms per horizon
  - Overlay forecast confidence bands on a 48-hour KPI time-series window

- [ ] 8. Optional LSTM classifier
  - Implement `src/models/lstm.py`
  - Implement `SLAViolationLSTM`: 2-layer LSTM (hidden=128, dropout=0.2) → FC(128→64→32→1) → Sigmoid
  - Implement `SLASequenceDataset`: sliding window with seq_len=24, stride=1, no shuffling
  - Use the same temporal splits as the XGBoost classifier
  - Save checkpoint to `models/lstm_{slice_type}_{horizon}min.pt`
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 9. Checkpoint — verify all models train and save correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Evaluation framework
  - [ ] 10.1 Implement `src/evaluation/evaluate.py` with all 8 pillars
    - Pillar 1 `verify_temporal_integrity`: assert no split overlap and violation rate in (0.01, 0.15) per split
    - Pillar 2 `compute_classification_metrics`: precision, recall, F1, AUC-ROC, AUC-PR, confusion matrix
    - Pillar 3 `compute_per_slice_metrics`: run Pillar 2 metrics per slice type
    - Pillar 4 `compute_per_event_recall`: recall per event_type label
    - Pillar 5 `compute_horizon_f1`: F1 at 15min, 30min, 60min
    - Pillar 6 `compute_lead_time_stats`: median, mean, P25, P75 lead times for true positives
    - Pillar 7 `compute_shap_importance`: `shap.TreeExplainer` only; save global summary bar plot to `reports/figures/`
    - Pillar 8 `compute_baseline_comparison`: static threshold baseline (alert when KPI within 15% of SLA threshold)
    - Implement `run_evaluation` orchestrator; Pillar 1 must pass before any other pillar runs
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 4.6, 12.5, 12.6_

  - [ ]* 10.2 Write unit tests for the evaluator
    - Test that `verify_temporal_integrity` raises an assertion error when splits overlap
    - Test that `compute_baseline_comparison` produces a result for each slice type
    - _Requirements: 8.1, 8.8_

- [ ]* 10.N Notebook — Full evaluation report (`notebooks/07_evaluation.ipynb`)
  - Run `run_evaluation()` for all 9 classifier models and display the nested results dict
  - Render per-slice precision / recall / F1 / AUC-PR as a grouped bar chart
  - Plot per-event-type recall as a horizontal bar chart
  - Show F1 vs. horizon line chart (15 / 30 / 60 min)
  - Display lead-time distribution as a box plot (median, P25, P75)
  - Render the SHAP global summary bar plot inline
  - Show XGBoost vs. static-threshold baseline comparison table

- [ ] 11. FastAPI REST API
  - [ ] 11.1 Implement `src/deployment/api.py`
    - Define `KPIHistory` and `PredictionResponse` Pydantic models
    - Implement `POST /predict`: validate ≥12 timesteps (HTTP 422 otherwise), build feature vector, run all 3 horizon classifiers, run forecasters, compute top-5 SHAP features, generate recommendations, derive health_status
    - Implement `GET /health/{slice_type}`: return health status; HTTP 404 for unknown slice_type
    - Implement `GET /slices`: return list of supported slice types
    - Load all 9 classifier models + thresholds at startup; load forecasters on demand
    - Implement recommendation engine mapping SHAP features to predefined recommendation strings
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ]* 11.2 Write unit tests for the API
    - Test that `/predict` with fewer than 12 timesteps returns HTTP 422
    - Test that `/predict` with an unknown slice_type returns HTTP 404
    - Test that `health_status` is "healthy" / "warning" / "critical" based on 30min probability thresholds
    - _Requirements: 9.2, 9.4, 9.8_

- [ ] 12. Streamlit dashboard
  - Implement `src/deployment/dashboard.py` with 6 pages
  - Page 1 (Slice Overview): risk gauges (green <30%, yellow 30–70%, red >70%), KPI delta indicators, side-by-side 3-slice comparison
  - Page 2 (Real-time Monitoring): KPI time-series plots with SLA threshold lines overlaid
  - Page 3 (Violation Prediction): probability timeline + KPI values + actual violations
  - Page 4 (Model Performance): confusion matrices, per-event recall, lead time distribution
  - Page 5 (Batch Analysis): CSV upload → backtesting via API `/predict`
  - Page 6 (Feature Importance): SHAP summary bar plots + individual waterfall explanations
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

- [ ] 13. Docker containerization
  - Create `Dockerfile.api`: install dependencies, copy `src/` and `models/`, expose port 8000, entrypoint `uvicorn src.deployment.api:app`
  - Create `Dockerfile.dashboard`: install dependencies, copy `src/`, expose port 8501, entrypoint `streamlit run src/deployment/dashboard.py`
  - Create `docker-compose.yml`: define `api` and `dashboard` services, mount `models/` into api container, set `API_URL` env var on dashboard, add healthcheck on `/slices`, declare `dashboard` depends_on `api`
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 14. Final checkpoint — full pipeline integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- The design document does not include a Correctness Properties section, so property-based tests are not included; unit tests cover correctness instead
- Checkpoints ensure incremental validation at the end of each major phase
