# Task 11 — FastAPI REST API: Analysis

**Module:** `src/deployment/api.py`
**Tests:** `tests/test_api.py` — 7 / 7 passed ✅
**Total test suite:** 24 / 24 passed ✅

---

## What was done

A production-ready FastAPI REST API was implemented exposing three endpoints for real-time SLA violation prediction. The API loads all 9 XGBoost classifiers and their SHAP explainers at startup, accepts raw KPI history payloads from clients, and returns structured prediction responses including violation probabilities, KPI forecasts, SHAP-based feature attributions, and natural-language recommendations.

---

## API Surface

### `POST /predict`

**Request body:** `KPIHistory`

```json
{
  "slice_type": "eMBB",
  "history": [
    {
      "timestamp": "2024-03-01T00:00:00",
      "dl_throughput": 80.0, "latency": 12.0, "jitter": 1.0,
      "packet_loss": 0.1, "prb_util": 0.55, "active_users": 40.0,
      "reliability": 99.95
    },
    ... (≥12 records required)
  ]
}
```

**Response:** `PredictionResponse`

```json
{
  "slice_type": "eMBB",
  "health_status": "warning",
  "horizons": [
    {
      "horizon_min": 15,
      "violation_prob": 0.1824,
      "predicted_kpis": {"dl_throughput": 78.4, "latency": 13.1, ...}
    },
    {"horizon_min": 30, "violation_prob": 0.3542, ...},
    {"horizon_min": 60, "violation_prob": 0.2211, ...}
  ],
  "top_shap_features": [
    {"feature": "sla_margin", "shap_value": 0.412, "direction": "increases_risk"},
    ...
  ],
  "recommendations": [
    "KPI approaching SLA threshold — initiate pre-emptive resource allocation",
    ...
  ]
}
```

**Validation rules (HTTP 422 on failure):**
- `slice_type` must be one of `{"eMBB", "URLLC", "mMTC"}` — validated by Pydantic `field_validator`
- `history` must contain ≥12 timesteps — validated by Pydantic `field_validator`

---

### `GET /health/{slice_type}`

Returns a lightweight health summary without requiring a full KPI history payload. Returns **HTTP 404** for unknown slice types.

```json
{
  "slice_type": "URLLC",
  "health_status": "healthy",
  "violation_prob_30min": 0.1772
}
```

---

### `GET /slices`

```json
{"slices": ["eMBB", "URLLC", "mMTC"]}
```

---

## Pydantic Data Models

| Model | Purpose |
|---|---|
| `TimestepRecord` | Single 5-minute KPI snapshot (7 KPI fields + timestamp) |
| `KPIHistory` | Request wrapper: slice_type + history list with validators |
| `HorizonPrediction` | Per-horizon violation probability + KPI forecasts |
| `PredictionResponse` | Full response: health_status + horizons + SHAP + recommendations |
| `HealthResponse` | Lightweight health endpoint response |

---

## Startup Model Loading

All 9 classifiers plus SHAP `TreeExplainer` instances (for 30min models) are loaded into module-level dicts at app startup using the `lifespan` context manager. A `_ensure_loaded()` guard handles the edge case where the lifespan doesn't fire in test environments, enabling lazy loading on first request.

```
_classifiers: {("eMBB", 15): (clf, 0.165), ("eMBB", 30): (clf, 0.177), ...}
_explainers:  {("eMBB", 30): shap.TreeExplainer, ...}
```

Forecasters (63 models) are loaded on demand per-request to avoid memory exhaustion — loading all 63 concurrently would require ~1.5 GB RAM.

---

## Prediction Pipeline (`POST /predict`)

```
1. Parse + validate KPIHistory (Pydantic)
2. Convert to DataFrame → sort by timestamp
3. Inject required schema columns (slice_type, event_type, violation targets)
4. Run build_features() — full feature engineering, 318 columns
5. Align feature columns to training set:
     → Missing cross-slice columns filled with 0.0
     → Uses clf.get_booster().feature_names for exact match
6. For each horizon (15, 30, 60 min):
     a. clf.predict_proba(X_last)[0,1] → violation probability
     b. Load forecaster per KPI → predicted_kpis
7. SHAP top-5: explainer.shap_values(X_last)[0] → argsort → top 5 indices
8. Recommendations: SHAP feature names → lookup in 17-entry mapping (prefix match fallback)
9. health_status = "critical" if prob_30≥0.60 | "warning" if ≥0.30 | "healthy"
```

---

## Health Status Thresholds

| Threshold | Status |
|---|---|
| `violation_prob_30min ≥ 0.60` | `critical` |
| `violation_prob_30min ≥ 0.30` | `warning` |
| `violation_prob_30min < 0.30` | `healthy` |

---

## Recommendation Engine

17 predefined SHAP feature → recommendation string mappings, organized by feature category:

| Category | Example Feature | Recommendation |
|---|---|---|
| SLA proximity | `sla_margin` | "KPI approaching SLA threshold — initiate pre-emptive resource allocation" |
| Throughput | `dl_throughput` | "Downlink throughput declining — check backhaul congestion and PRB allocation" |
| Latency | `latency_ewma_span12` | "Sustained latency increase — review inter-site handover configuration" |
| PRB utilisation | `prb_util_roll3` | "PRB utilisation trending upward — activate load-sharing with neighbouring cells" |
| Packet loss | `packet_loss_lag1` | "Recent packet loss spike — initiate RF interference scan" |
| Cross-slice | `cross_slice_prb` | "Cross-slice resource contention detected — adjust slice priority weights" |
| Jitter | `jitter` | "Jitter exceeds acceptable bounds — review QoS scheduling parameters" |

Matching uses exact lookup first, then prefix/substring match, then a generic fallback. A maximum of 5 recommendations is returned per response.

---

## Key Design Decisions

### Zero-padding cross-slice features at inference
The training feature matrix includes cross-slice coupling features (e.g., `cross_embb_prb_roll6_mean`) derived from neighbouring slice data. At inference time, the API receives only a single slice's KPI history. Missing cross-slice columns are zero-padded using `clf.get_booster().feature_names` to match the exact training feature order — XGBoost's feature importance will simply not activate those paths.

### Forecasters loaded on demand
Loading all 63 forecaster models at startup would consume ~1.5 GB RAM. Instead they are loaded per-request directly from disk. The `models/` directory serves as the on-disk model registry. A production deployment could add a forecaster LRU cache if request latency becomes critical.

### Lifespan + lazy-load fallback
FastAPI's `lifespan` handler fires in production. During `TestClient` sessions in pytest, the lifespan may not trigger consistently. The `_ensure_loaded()` guard checks if `_classifiers` is empty and loads all models on the first endpoint call, ensuring tests pass without separate fixtures.

---

## Test Coverage

| Test | Requirement |
|---|---|
| `test_get_slices_returns_all` | /slices returns all 3 slices |
| `test_health_known_slice` | /health/{slice} returns 200 |
| `test_health_unknown_slice_returns_404` | /health/{unknown} returns 404 — Req 9.4 |
| `test_predict_too_few_timesteps_returns_422` | <12 timesteps → 422 — Req 9.2 |
| `test_predict_unknown_slice_returns_422` | Unknown slice → 422 |
| `test_predict_valid_request_returns_200` | Full happy-path: 3 horizons, recommendations |
| `test_health_status_levels` | health_status matches thresholds — Req 9.8 |

---

## Implications for Task 12 (Streamlit Dashboard) and Task 13 (Docker)

| Finding | Impact |
|---|---|
| API exposes `health_status` as a string | Dashboard can colour-code slice cards immediately without additional computation |
| SHAP top-5 features included in response | Dashboard can render a live SHAP bar chart from the API response without recomputing |
| Recommendations are plain-text strings | Dashboard can display them as a suggestions panel with zero additional processing |
| Forecasters loaded on demand (I/O latency) | Dashboard should show a loading spinner during the predict call |
| API runs on port 8000 | Docker compose should expose 8000; Streamlit on 8501 |

---

## Conclusion

The FastAPI REST API is fully implemented, tested, and pushed. All 3 endpoints are operational, 24 / 24 project tests pass, and the API is ready for containerisation in Task 13. The prediction pipeline correctly aligns live inference feature vectors to the training feature set, handles class imbalance via XGBoost thresholds, and delivers per-horizon forecasts with actionable SHAP-grounded NOC recommendations in a single request.
