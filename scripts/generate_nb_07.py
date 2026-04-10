import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Task 10 — Full Evaluation Report\n\n"
    "Runs all 8 evaluation pillars across the 9 XGBoost classifiers and renders "
    "per-slice, per-event, per-horizon, and SHAP-based insights."
))

cells.append(nbf.v4.new_code_cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.evaluation.evaluate import run_evaluation

%matplotlib inline
plt.rcParams.update({"figure.dpi": 110})
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Run full evaluation"))

cells.append(nbf.v4.new_code_cell("""\
results = run_evaluation(
    raw_dir=Path("../data/raw/generated"),
    models_dir=Path("../models"),
    figures_dir=Path("../reports/figures"),
    verbose=True,
)
# Print all results except the SHAP dict (too long)
print(json.dumps({k: v for k, v in results.items() if k != "p7_shap"}, indent=2))
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Per-slice Precision / Recall / F1 / AUC-PR (30min)"))

cells.append(nbf.v4.new_code_cell("""\
slices = ["eMBB", "URLLC", "mMTC"]
metrics = ["precision", "recall", "f1", "auc_pr"]
colors  = ["#4C9BE8", "#F5A623", "#7ED321", "#B86CFF"]

x = np.arange(len(slices))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 5))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [results["p3_per_slice"][s][metric] for s in slices]
    bars = ax.bar(x + i * width, vals, width, label=metric.upper(), color=color)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(slices, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Per-Slice Classification Metrics — 30min Horizon")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Per-event-type recall (eMBB 30min)"))

cells.append(nbf.v4.new_code_cell("""\
event_recall = results["p4_event_recall"]
fig, ax = plt.subplots(figsize=(9, 4))
events = list(event_recall.keys())
vals   = [event_recall[e] for e in events]
colors_evt = ["#4C9BE8" if v > 0.5 else "#E84C4C" for v in vals]
ax.barh(events, vals, color=colors_evt)
ax.axvline(0.9, color="black", linestyle="--", linewidth=1, label="0.90 recall target")
ax.set_xlim(0, 1.1)
ax.set_xlabel("Recall")
ax.set_title("Per-Event-Type Recall — eMBB 30min")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. F1 vs horizon (all slices)"))

cells.append(nbf.v4.new_code_cell("""\
horizons = [15, 30, 60]
fig, ax = plt.subplots(figsize=(8, 4))
colors_slice = {"eMBB": "#4C9BE8", "URLLC": "#F5A623", "mMTC": "#7ED321"}

for stype in ["eMBB", "URLLC", "mMTC"]:
    f1s = [results["p5_horizon_f1"][stype][h] for h in horizons]
    ax.plot(horizons, f1s, marker="o", label=stype, color=colors_slice[stype], linewidth=2)

ax.set_xticks(horizons)
ax.set_xticklabels(["15 min", "30 min", "60 min"])
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs Prediction Horizon")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Lead-time distribution (box plot)"))

cells.append(nbf.v4.new_code_cell("""\
lead = results["p6_lead_time"]
fig, ax = plt.subplots(figsize=(8, 4))
slices = ["eMBB", "URLLC", "mMTC"]
colors_lead = ["#4C9BE8", "#F5A623", "#7ED321"]

for i, (stype, color) in enumerate(zip(slices, colors_lead)):
    lt = lead[stype]
    if lt["median"] is None:
        continue
    ax.boxplot(
        x=[[lt["p25"], lt["median"], lt["p75"]]],  # fake box from quartiles
        positions=[i],
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
    )
    ax.text(i, lt["p75"] + 0.5, f"n={lt['n_tp']}", ha="center", fontsize=9)

ax.set_xticks(range(len(slices)))
ax.set_xticklabels(slices)
ax.set_ylabel("Lead time (minutes)")
ax.set_title("True-Positive Lead-Time Distribution — 30min Horizon")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. SHAP global summary bar plot"))

cells.append(nbf.v4.new_code_cell("""\
shap_scores = results["p7_shap"]
top_feats  = list(shap_scores.keys())[:20]
top_vals   = [shap_scores[f] for f in top_feats]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_feats[::-1], top_vals[::-1], color="#4C9BE8")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP Global Feature Importance — eMBB 30min (from saved results)")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. XGBoost vs static-threshold baseline"))

cells.append(nbf.v4.new_code_cell("""\
baseline = results["p8_baseline"]
rows = []
for stype in ["eMBB", "URLLC", "mMTC"]:
    for model in ["xgboost", "baseline"]:
        m = baseline[stype][model]
        rows.append({
            "Slice": stype, "Model": model.capitalize(),
            "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"]
        })

df_cmp = pd.DataFrame(rows)
print(df_cmp.to_string(index=False))
"""))

nb.cells = cells
out = Path("notebooks/07_evaluation.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {out}")
