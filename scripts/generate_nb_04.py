import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("# Task 4 — Temporal Data Splitting\n\nValidates that the 90-day feature matrices are partitioned into train / val / test with **zero timestamp overlap** and that violation rates are stable across splits."))

cells.append(nbf.v4.new_code_cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

%matplotlib inline
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (14, 4)})

from src.data.splitter import temporal_split
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Run temporal split on eMBB features"))

cells.append(nbf.v4.new_code_cell("""\
# Load the engineered-feature Parquet (built by Task 3)
# Falls back to raw synthetic if processed is not available
processed = Path("../data/processed/eMBB.parquet")
raw       = Path("../data/raw/generated/embb_synthetic.parquet")
src_path  = processed if processed.exists() else raw

print(f"Loading from: {src_path}")
df = pd.read_parquet(src_path)
print(f"Total rows : {len(df):,}")
print(f"Date range : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

train, val, test = temporal_split(df)

for name, split in [("Train", train), ("Val", val), ("Test", test)]:
    rate = split["violation_in_30min"].mean() * 100 if "violation_in_30min" in split.columns else float("nan")
    print(f"{name:5s}  rows={len(split):6,}  "
          f"{split['timestamp'].min().date()} → {split['timestamp'].max().date()}  "
          f"violation_30min={rate:.2f}%")
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Timeline bar — split boundaries"))

cells.append(nbf.v4.new_code_cell("""\
colors = {"Train": "#4C9BE8", "Val": "#F5A623", "Test": "#7ED321"}
splits = {"Train": train, "Val": val, "Test": test}

fig, ax = plt.subplots(figsize=(14, 2))

for label, split in splits.items():
    t0 = split["timestamp"].min()
    t1 = split["timestamp"].max()
    ax.barh(0, (t1 - t0).days, left=(t0 - train["timestamp"].min()).days,
            height=0.6, color=colors[label], label=label, edgecolor="white")
    mid = (t0 - train["timestamp"].min()).days + (t1 - t0).days / 2
    ax.text(mid, 0, f"{label}\\n{len(split):,} rows", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

ax.set_yticks([])
ax.set_xlabel("Day")
ax.set_title("Temporal Split Boundaries — eMBB Slice")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Violation rate per split"))

cells.append(nbf.v4.new_code_cell("""\
if "violation_in_30min" in df.columns:
    labels = ["Train", "Val", "Test"]
    rates  = [s["violation_in_30min"].mean() * 100 for s in [train, val, test]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, rates, color=list(colors.values()), edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)
    ax.set_ylabel("Positive rate (%)")
    ax.set_title("violation_in_30min Positive Rate per Split")
    ax.set_ylim(0, max(rates) * 1.4)
    plt.tight_layout()
    plt.show()
else:
    print("violation_in_30min column not found — skipping rate chart.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. KPI time-series with split boundaries"))

cells.append(nbf.v4.new_code_cell("""\
kpi = "latency" if "latency" in df.columns else df.select_dtypes("number").columns[0]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["timestamp"], df[kpi], color="#888", linewidth=0.5, alpha=0.7, label=kpi)

for label, split in splits.items():
    ax.axvline(split["timestamp"].min(), color=colors[label], linewidth=1.5,
               linestyle="--", label=f"{label} start")

ax.set_title(f"{kpi} with Temporal Split Boundaries")
ax.set_xlabel("Timestamp")
ax.set_ylabel(kpi)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""))

nb.cells = cells

out = Path("notebooks/04_temporal_split.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {out}")
