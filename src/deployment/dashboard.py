"""
Streamlit Dashboard — src/deployment/dashboard.py

6-page dashboard for the 5G QoS SLA Violation Predictor.

Pages
-----
1. Slice Overview       — risk gauges, KPI delta indicators, 3-slice comparison
2. Real-time Monitoring — KPI time-series with SLA threshold lines
3. Violation Prediction — probability timeline + actual violations
4. Model Performance    — confusion matrices, per-event recall, lead-time distribution
5. Batch Analysis       — CSV upload → backtesting via API /predict
6. Feature Importance   — SHAP summary bars + waterfall explanation

Run with:
    streamlit run src/deployment/dashboard.py

API_URL env var controls where the API is reached (default: http://localhost:8000).
"""

from __future__ import annotations

import os
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")

SLICE_COLORS = {"eMBB": "#4C9BE8", "URLLC": "#F5A623", "mMTC": "#7ED321"}
SLICE_TYPES  = ["eMBB", "URLLC", "mMTC"]

SLA_THRESHOLDS = {
    "eMBB":  {"dl_throughput": 50.0,  "latency": 30.0,  "packet_loss": 1.0},
    "URLLC": {"dl_throughput": 10.0,  "latency": 5.0,   "packet_loss": 0.1},
    "mMTC":  {"dl_throughput": 1.0,   "latency": 500.0, "packet_loss": 3.0},
}

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="5G QoS Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS injected once
st.markdown("""
<style>
  .metric-card {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border-radius: 12px;
      padding: 1rem 1.4rem;
      border-left: 4px solid #4C9BE8;
      margin-bottom: 0.6rem;
  }
  .gauge-critical { color: #FF4B4B; font-weight: 700; }
  .gauge-warning  { color: #FFA500; font-weight: 700; }
  .gauge-healthy  { color: #21C55D; font-weight: 700; }
  .page-title { font-size: 1.7rem; font-weight: 700; margin-bottom: 0.3rem; }
  .subtitle   { color: #888; font-size: 0.9rem; margin-bottom: 1.2rem; }
</style>""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _api_health(slice_type: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health/{slice_type}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _api_predict(slice_type: str, history: list[dict]) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"slice_type": slice_type, "history": history},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _health_color(status: str) -> str:
    return {"critical": "#FF4B4B", "warning": "#FFA500", "healthy": "#21C55D"}.get(status, "#888")


def _health_emoji(status: str) -> str:
    return {"critical": "🔴", "warning": "🟡", "healthy": "🟢"}.get(status, "⚪")


def _gauge_figure(prob: float, label: str, color: str) -> plt.Figure:
    """Draw a semicircular gauge for violation probability."""
    fig, ax = plt.subplots(figsize=(3, 1.8), subplot_kw=dict(polar=False))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.axis("off")

    theta_range = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta_range), np.sin(theta_range), color="#333", lw=8, solid_capstyle="round")

    filled = np.linspace(0, np.pi * prob, 200)
    ax.plot(np.cos(filled), np.sin(filled), color=color, lw=8, solid_capstyle="round")

    ax.text(0, 0.35, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=18, color="white", fontweight="bold")
    ax.text(0, -0.05, label, ha="center", va="center",
            fontsize=8, color="#aaa")
    return fig


def _make_synthetic_history(slice_type: str, n: int = 48,
                             jitter_scale: float = 1.0) -> list[dict]:
    """Generate n synthetic KPI timesteps for demo purposes."""
    base = datetime.now() - timedelta(minutes=5 * n)
    defaults = {
        "eMBB":  dict(dl_throughput=80.0, latency=12.0, jitter=1.0,
                      packet_loss=0.1, prb_util=0.55, active_users=40.0, reliability=99.95),
        "URLLC": dict(dl_throughput=25.0, latency=2.5, jitter=0.3,
                      packet_loss=0.02, prb_util=0.40, active_users=10.0, reliability=99.999),
        "mMTC":  dict(dl_throughput=2.0, latency=200.0, jitter=10.0,
                      packet_loss=0.5, prb_util=0.60, active_users=2000.0, reliability=99.8),
    }[slice_type]
    rng = np.random.default_rng(seed=42)
    history = []
    for i in range(n):
        ts = (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S")
        row = {"timestamp": ts}
        for kpi, base_val in defaults.items():
            noise = rng.normal(0, base_val * 0.05 * jitter_scale)
            row[kpi] = max(0.0, round(base_val + noise, 4))
        history.append(row)
    return history


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/network.png", width=60)
    st.markdown("## **5G QoS Predictor**")
    st.markdown("Real-time SLA violation detection")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊 Slice Overview",
            "📡 Real-time Monitoring",
            "🔮 Violation Prediction",
            "📈 Model Performance",
            "🗂 Batch Analysis",
            "🧠 Feature Importance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    selected_slice = st.selectbox("Active Slice", SLICE_TYPES)
    horizon = st.selectbox("Horizon (min)", [15, 30, 60], index=1)
    st.markdown(f"**API:** `{API_URL}`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SLICE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Slice Overview":
    st.markdown('<p class="page-title">📊 Slice Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Risk gauges and KPI delta indicators across all 3 slices</p>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    for col, stype in zip(cols, SLICE_TYPES):
        with col:
            health = _api_health(stype)
            if health:
                prob  = health["violation_prob_30min"]
                status = health["health_status"]
                color = _health_color(status)
                fig = _gauge_figure(prob, f"{stype} 30min", color)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f'<div style="text-align:center;font-size:1.1rem;font-weight:700;color:{color}">'
                    f'{_health_emoji(status)} {status.upper()}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"{stype}: API unreachable")

    st.divider()
    st.subheader("KPI Comparison Across Slices")

    # Fetch synthetic predictions for each slice for the comparison table
    comparison_rows = []
    for stype in SLICE_TYPES:
        hist = _make_synthetic_history(stype, n=48)
        pred = _api_predict(stype, hist)
        if pred:
            h30 = next((h for h in pred["horizons"] if h["horizon_min"] == 30), None)
            if h30 and h30["predicted_kpis"]:
                row = {"Slice": stype, "Status": pred["health_status"].capitalize()}
                for k in ["dl_throughput", "latency", "packet_loss", "prb_util"]:
                    row[k] = round(h30["predicted_kpis"].get(k, float("nan")), 3)
                comparison_rows.append(row)

    if comparison_rows:
        df_cmp = pd.DataFrame(comparison_rows)
        st.dataframe(df_cmp.set_index("Slice"), use_container_width=True)
    else:
        st.info("Start the API server to see live predictions.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REAL-TIME MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Real-time Monitoring":
    st.markdown('<p class="page-title">📡 Real-time Monitoring</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">KPI time-series with SLA threshold lines overlaid</p>',
                unsafe_allow_html=True)

    kpi_select = st.multiselect(
        "KPIs to display",
        ["dl_throughput", "latency", "jitter", "packet_loss", "prb_util"],
        default=["dl_throughput", "latency"],
    )
    n_points = st.slider("History window (timesteps)", 24, 288, 96)
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)

    hist = _make_synthetic_history(selected_slice, n=n_points, jitter_scale=1.5)
    df_hist = pd.DataFrame(hist)
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

    thresholds = SLA_THRESHOLDS.get(selected_slice, {})

    if kpi_select:
        n_kpis = len(kpi_select)
        fig, axes = plt.subplots(n_kpis, 1, figsize=(14, 3 * n_kpis), sharex=True)
        if n_kpis == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0e1117")

        for ax, kpi in zip(axes, kpi_select):
            ax.set_facecolor("#0e1117")
            color = SLICE_COLORS[selected_slice]
            ax.plot(df_hist["timestamp"], df_hist[kpi], color=color, lw=1.5, label=kpi)
            ax.fill_between(df_hist["timestamp"], df_hist[kpi], alpha=0.15, color=color)

            if kpi in thresholds:
                thr = thresholds[kpi]
                ax.axhline(thr, color="#FF4B4B", linestyle="--", lw=1.2,
                           label=f"SLA: {thr}")
                ax.fill_between(df_hist["timestamp"], thr, df_hist[kpi].max() * 1.1,
                                alpha=0.05, color="#FF4B4B")

            ax.set_ylabel(kpi, color="#aaa", fontsize=9)
            ax.tick_params(colors="#aaa", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.legend(fontsize=8, loc="upper right",
                      facecolor="#0e1117", labelcolor="white")

        plt.xticks(rotation=30, ha="right", color="#aaa", fontsize=7)
        plt.suptitle(f"{selected_slice} KPI Monitoring", color="white", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if auto_refresh:
        time.sleep(30)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIOLATION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Violation Prediction":
    st.markdown('<p class="page-title">🔮 Violation Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Violation probability timeline and top recommendations</p>',
                unsafe_allow_html=True)

    n_points = st.slider("Simulation timesteps", 48, 288, 96)
    jitter   = st.slider("Traffic variability", 0.5, 3.0, 1.0, step=0.5)

    if st.button("▶  Run Prediction", type="primary") or "pred_result" not in st.session_state:
        hist = _make_synthetic_history(selected_slice, n=n_points, jitter_scale=jitter)
        with st.spinner("Calling API…"):
            pred = _api_predict(selected_slice, hist)
        st.session_state["pred_result"] = pred
        st.session_state["pred_hist"]   = hist
    else:
        pred = st.session_state.get("pred_result")
        hist = st.session_state.get("pred_hist", [])

    if pred:
        st.success(f"Health status: {_health_emoji(pred['health_status'])} **{pred['health_status'].upper()}**")

        # Probability bar chart per horizon
        horizons  = [h["horizon_min"] for h in pred["horizons"]]
        probs     = [h["violation_prob"] for h in pred["horizons"]]
        bar_colors= ["#21C55D" if p < 0.30 else "#FFA500" if p < 0.60 else "#FF4B4B"
                     for p in probs]

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        bars = ax.bar([f"{h} min" for h in horizons], probs, color=bar_colors, width=0.5)
        ax.bar_label(bars, fmt="%.2f", label_type="edge", color="white", fontsize=11)
        ax.axhline(0.30, color="#FFA500", linestyle="--", lw=1, label="Warning (0.30)")
        ax.axhline(0.60, color="#FF4B4B", linestyle="--", lw=1, label="Critical (0.60)")
        ax.set_ylabel("Violation Probability", color="#aaa")
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
        plt.title(f"{selected_slice} — SLA Violation Probability by Horizon", color="white")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # KPI forecasts table
        h_sel = next((h for h in pred["horizons"] if h["horizon_min"] == horizon), pred["horizons"][0])
        if h_sel["predicted_kpis"]:
            st.subheader(f"Predicted KPIs at +{h_sel['horizon_min']} min")
            kpi_df = pd.DataFrame([
                {"KPI": k, "Forecast": v,
                 "SLA Limit": SLA_THRESHOLDS.get(selected_slice, {}).get(k, "—")}
                for k, v in h_sel["predicted_kpis"].items()
            ])
            st.dataframe(kpi_df.set_index("KPI"), use_container_width=True)

        # Recommendations
        if pred["recommendations"]:
            st.subheader("💡 NOC Recommendations")
            for i, rec in enumerate(pred["recommendations"], 1):
                st.markdown(f"**{i}.** {rec}")
    else:
        st.info("Start the API and click **Run Prediction** to see results.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<p class="page-title">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Confusion matrices, per-event recall, and lead-time distribution</p>',
                unsafe_allow_html=True)

    @st.cache_data(ttl=300)
    def _load_eval_results():
        try:
            sys_path_fix = str(Path(__file__).resolve().parent.parent.parent)
            import sys; sys.path.insert(0, sys_path_fix)
            from src.evaluation.evaluate import run_evaluation
            return run_evaluation(verbose=False)
        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Loading evaluation results…"):
        ev = _load_eval_results()

    if "error" in ev:
        st.error(f"Evaluation failed: {ev['error']}")
    else:
        # ── Per-slice metrics bar chart ────────────────────────────────────
        st.subheader("Per-Slice Classification Metrics (30min horizon)")
        metrics = ["precision", "recall", "f1", "auc_pr"]
        x = np.arange(len(SLICE_TYPES))
        width = 0.18
        colors_m = ["#4C9BE8", "#F5A623", "#7ED321", "#B86CFF"]

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        for i, (m, c) in enumerate(zip(metrics, colors_m)):
            vals = [ev["p3_per_slice"][s][m] for s in SLICE_TYPES]
            ax.bar(x + i * width, vals, width, label=m.upper(), color=c)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(SLICE_TYPES, color="#aaa")
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
        plt.title("Classification Metrics — All Slices", color="white")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Per-event recall ────────────────────────────────────────────────
        if ev.get("p4_event_recall"):
            st.subheader("Per-Event-Type Recall (eMBB 30min)")
            er = ev["p4_event_recall"]
            events = list(er.keys())
            vals = [er[e] for e in events]
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            bar_colors = ["#21C55D" if v >= 0.90 else "#FFA500" if v > 0 else "#FF4B4B"
                          for v in vals]
            ax.barh(events, vals, color=bar_colors)
            ax.axvline(0.90, color="white", linestyle="--", lw=1, label="0.90 target")
            ax.set_xlim(0, 1.1)
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#333")
            ax.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
            plt.title("Per-Event Recall", color="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Lead-time distribution ──────────────────────────────────────────
        if ev.get("p6_lead_time"):
            st.subheader("True-Positive Lead-Time Distribution (30min horizon)")
            lt = ev["p6_lead_time"]
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            for i, stype in enumerate(SLICE_TYPES):
                d = lt[stype]
                if d["median"] is not None:
                    ax.barh(stype, d["p75"] - d["p25"],
                            left=d["p25"], height=0.4, color=SLICE_COLORS[stype], alpha=0.7)
                    ax.plot([d["median"]], [stype], "w|", ms=12, mew=2)
                    ax.text(d["p75"] + 0.5, stype, f"n={d['n_tp']}", va="center",
                            color="#aaa", fontsize=9)
            ax.set_xlabel("Lead time (minutes)", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#333")
            plt.title("Lead-Time IQR (P25–P75) with Median", color="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗂 Batch Analysis":
    st.markdown('<p class="page-title">🗂 Batch Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a CSV of KPI snapshots to backtest via the API</p>',
                unsafe_allow_html=True)

    st.info(
        "CSV must contain columns: `timestamp, dl_throughput, latency, jitter, "
        "packet_loss, prb_util, active_users, reliability`"
    )

    uploaded = st.file_uploader("Upload KPI CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_up)}** rows")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("▶  Run Batch Prediction", type="primary"):
            required = {"timestamp","dl_throughput","latency","jitter",
                        "packet_loss","prb_util","active_users","reliability"}
            if not required.issubset(df_up.columns):
                st.error(f"Missing columns: {required - set(df_up.columns)}")
            elif len(df_up) < 12:
                st.error("Need at least 12 rows for feature engineering.")
            else:
                history = df_up[list(required)].to_dict(orient="records")
                with st.spinner("Calling API…"):
                    pred = _api_predict(selected_slice, history)

                if pred:
                    st.success(f"health_status: **{pred['health_status'].upper()}**")
                    rows = []
                    for h in pred["horizons"]:
                        rows.append({
                            "Horizon": f"{h['horizon_min']}min",
                            "Violation Prob": h["violation_prob"],
                        })
                    st.table(pd.DataFrame(rows).set_index("Horizon"))
                    if pred["recommendations"]:
                        st.subheader("💡 Recommendations")
                        for r in pred["recommendations"]:
                            st.markdown(f"• {r}")
                else:
                    st.error("API returned an error. Check that the server is running.")
    else:
        # Show example template
        example = _make_synthetic_history("eMBB", n=5)
        df_ex = pd.DataFrame(example)
        st.subheader("Example CSV format")
        st.dataframe(df_ex, use_container_width=True)
        csv_bytes = df_ex.to_csv(index=False).encode()
        st.download_button("⬇ Download example CSV", csv_bytes,
                           file_name="example_kpi.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Feature Importance":
    st.markdown('<p class="page-title">🧠 Feature Importance</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">SHAP global summary and per-prediction waterfall explanation</p>',
                unsafe_allow_html=True)

    # ── Global saved SHAP figure ───────────────────────────────────────────
    shap_fig_path = Path("reports/figures") / f"shap_{selected_slice.lower()}_{horizon}min.png"
    if shap_fig_path.exists():
        st.subheader(f"Global SHAP Feature Importance — {selected_slice} {horizon}min")
        st.image(str(shap_fig_path), use_container_width=True)
    else:
        st.info("SHAP figure not found. Run `src/evaluation/evaluate.py` to generate it.")

    st.divider()

    # ── Live SHAP top-5 via API ────────────────────────────────────────────
    st.subheader("Live SHAP Attribution for Current Prediction")
    if st.button("▶  Compute Live SHAP", type="primary"):
        hist = _make_synthetic_history(selected_slice, n=48)
        with st.spinner("Calling API…"):
            pred = _api_predict(selected_slice, hist)

        if pred and pred.get("top_shap_features"):
            shap_data = pred["top_shap_features"]
            feats  = [d["feature"] for d in shap_data]
            shap_v = [d["shap_value"] for d in shap_data]
            dirs   = [d["direction"] for d in shap_data]

            bar_colors = ["#FF4B4B" if d == "increases_risk" else "#21C55D" for d in dirs]

            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            ax.barh(feats[::-1], shap_v[::-1], color=bar_colors[::-1])
            ax.axvline(0, color="white", lw=0.8)
            ax.set_xlabel("SHAP value (impact on log-odds)", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#333")

            legend_els = [
                mpatches.Patch(color="#FF4B4B", label="Increases risk"),
                mpatches.Patch(color="#21C55D", label="Decreases risk"),
            ]
            ax.legend(handles=legend_els, fontsize=8,
                      facecolor="#0e1117", labelcolor="white")
            plt.title(f"Top-5 SHAP Features — {selected_slice} 30min", color="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("#### 💡 Recommendations")
            for rec in pred.get("recommendations", []):
                st.markdown(f"• {rec}")
        else:
            st.warning("No SHAP data returned. Check API connectivity.")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#555;font-size:0.8rem">'
    '5G QoS SLA Violation Predictor • Powered by XGBoost + FastAPI + Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
