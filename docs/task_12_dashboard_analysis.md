# Task 12 — Streamlit Dashboard: Analysis

**Module:** `src/deployment/dashboard.py`
**Pages:** 6
**Entry point:** `streamlit run src/deployment/dashboard.py`
**API dependency:** `http://localhost:8000` (configurable via `API_URL` env var)

---

## What was done

A 6-page Streamlit dashboard was implemented as the primary visual interface for the 5G QoS SLA Violation Predictor system. The dashboard communicates exclusively with the FastAPI backend via HTTP — it does not load models directly — preserving clean separation between inference logic and presentation. All pages share a sidebar with a global slice selector and horizon control.

---

## Pages

### Page 1 — Slice Overview

**Purpose:** High-level health snapshot for all 3 slices simultaneously.

- **Semicircular risk gauges** — drawn with Matplotlib polar math, colour-coded:
  - 🟢 Green: < 30% violation probability
  - 🟡 Yellow: 30–60%
  - 🔴 Red: > 60%
- Calls `GET /health/{slice_type}` for each of eMBB, URLLC, mMTC simultaneously
- **KPI comparison table** — fetches `POST /predict` with 48-step synthetic history per slice, renders forecasted KPIs at the 30min horizon side-by-side

---

### Page 2 — Real-time Monitoring

**Purpose:** Time-series visualisation of raw KPI readings with SLA boundaries overlaid.

- Multi-KPI selector (dl_throughput, latency, jitter, packet_loss, prb_util)
- History window slider: 24–288 timesteps
- SLA threshold lines overlaid as horizontal dashed red lines with a subtle red fill zone above/below
- **Auto-refresh toggle** — calls `st.rerun()` every 30 seconds to simulate live data
- Dark-mode Matplotlib figures (facecolor `#0e1117`) integrated natively into Streamlit

---

### Page 3 — Violation Prediction

**Purpose:** Horizon-level violation probability with KPI forecasts and NOC recommendations.

- **Run Prediction** button → calls `POST /predict`, stores result in `st.session_state`
- **Probability bar chart** — 3 bars (15/30/60 min), individually coloured by threshold zone
- Warning threshold lines at 0.30 and 0.60 overlaid
- **KPI forecast table** — shows predicted values at selected horizon vs SLA limits
- **Recommendations panel** — renders up to 5 NOC action strings from the SHAP recommendation engine
- Traffic variability slider controls the synthetic history noise level for demo purposes

---

### Page 4 — Model Performance

**Purpose:** Evaluation framework results rendered visually.

- Calls `run_evaluation()` internally via `@st.cache_data(ttl=300)` to avoid re-running every render
- **Per-slice metrics bar chart** — Precision / Recall / F1 / AUC-PR across eMBB / URLLC / mMTC
- **Per-event-type recall bar chart** — horizontal bars, green ≥ 0.90, yellow > 0, red = 0
- **Lead-time IQR chart** — horizontal box-style bars from P25 to P75 with median marker

---

### Page 5 — Batch Analysis

**Purpose:** CSV-based backtesting for bulk historical KPI data.

- CSV upload widget with column validation (8 required columns)
- Validates: ≥12 rows required before calling API
- Submits full CSV as `history` to `POST /predict`
- Renders per-horizon violation probabilities as a table + recommendations
- **Example CSV download** — generates a 5-row synthetic example and offers it for download via `st.download_button`

---

### Page 6 — Feature Importance

**Purpose:** SHAP global and per-prediction attribution.

- **Global SHAP figure** — loads the pre-saved PNG from `reports/figures/shap_{slice}_{horizon}min.png` if available
- **Live SHAP panel** — calls `POST /predict` and renders the `top_shap_features` list as a waterfall-style horizontal bar chart
  - Red bars = `increases_risk`, green bars = `decreases_risk`
  - `axvline(0)` bisector line
- Companion recommendations panel below

---

## Design Decisions

### API-first architecture
The dashboard never imports model or evaluation code directly for live predictions. All inference routes through `POST /predict` and `GET /health/{slice}`. This means the dashboard works identically whether it runs locally, in Docker, or against a remote API endpoint.

### Session state for prediction persistence
`st.session_state["pred_result"]` caches the last API response across rerenders on Page 3. This prevents re-calling the API on every Streamlit interaction event (slider moves, etc.), which would be expensive given forecaster I/O.

### Evaluation page uses direct Python call (not API)
The `run_evaluation()` function is called directly from `src.evaluation.evaluate` rather than going through an API endpoint — the evaluation framework exposes no HTTP endpoint in the current spec. The result is cached for 5 minutes via `@st.cache_data(ttl=300)`.

### Dark-mode consistency
All Matplotlib figures are configured with `facecolor="#0e1117"` (Streamlit's default dark background) to prevent jarring white figure backgrounds in dark mode. Axes use `#aaa` tick colours and `#333` spine borders.

### Synthetic history generation
`_make_synthetic_history()` provides reproducible (seeded RNG) demo data for pages that don't require real uploaded data. The `jitter_scale` parameter on Page 3 lets users simulate degraded network conditions for demonstration purposes.

---

## API Calls Summary

| Page | Endpoint Called | Purpose |
|---|---|---|
| 1 — Slice Overview | `GET /health/{slice}` × 3 | Gauge probabilities |
| 1 — Slice Overview | `POST /predict` × 3 | KPI comparison table |
| 2 — Monitoring | None (local synthetic data) | KPI time-series only |
| 3 — Violation Prediction | `POST /predict` | Full prediction response |
| 4 — Model Performance | None (direct `run_evaluation()`) | Evaluation metrics |
| 5 — Batch Analysis | `POST /predict` | CSV backtesting |
| 6 — Feature Importance | `POST /predict` | Live SHAP top-5 |

---

## How to Run

```bash
# Terminal 1 — start the API (already running on port 8000)
venv/bin/python -m src.deployment.api

# Terminal 2 — start the dashboard
venv/bin/streamlit run src/deployment/dashboard.py
# → Opens http://localhost:8501
```

To point the dashboard at a different API host:
```bash
API_URL=http://192.168.1.10:8000 venv/bin/streamlit run src/deployment/dashboard.py
```

---

## Implications for Task 13 (Docker)

| Finding | Docker Impact |
|---|---|
| Dashboard reads `API_URL` from env | `docker-compose.yml` sets `API_URL=http://api:8000` |
| Pre-saved SHAP PNGs in `reports/figures/` | Mount `reports/` as a volume or bake into image |
| Evaluation loads models from `models/` | Mount `models/` into both containers |
| Streamlit default port 8501 | Expose 8501 on dashboard container |
| API port 8000 | Dashboard `depends_on: api` in compose |

---

## Conclusion

The 6-page Streamlit dashboard is fully implemented, syntactically validated, and pushed. It provides a complete visual front-end for the 5G QoS Predictor covering real-time health monitoring, violation prediction with horizon-level breakdowns, SHAP-grounded explainability, batch backtesting, and model performance inspection — all backed exclusively by the FastAPI service.
