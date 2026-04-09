import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Task 6 — XGBoost SLA Violation Classifier\n\n"
    "Trains 9 binary XGBoost classifiers (3 slices × 3 horizons) and evaluates them "
    "on the held-out test split."
))

cells.append(nbf.v4.new_code_cell("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    PrecisionRecallDisplay, RocCurveDisplay,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from src.models.classifier import (
    load_classifier, _prepare_X_y, _get_feature_cols,
    HORIZONS,
)
from src.features.engineering import build_features
from src.data.splitter import temporal_split

%matplotlib inline
plt.rcParams.update({"figure.dpi": 110})
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load data and prepare test splits"))

cells.append(nbf.v4.new_code_cell("""\
RAW = Path("../data/raw/generated")
slices_raw = {
    "eMBB":  pd.read_parquet(RAW / "embb_synthetic.parquet"),
    "URLLC": pd.read_parquet(RAW / "urllc_synthetic.parquet"),
    "mMTC":  pd.read_parquet(RAW / "mmtc_synthetic.parquet"),
}

test_splits = {}
for stype, df_raw in slices_raw.items():
    others = {k: v for k, v in slices_raw.items() if k != stype}
    df_feat = build_features(df_raw, stype, others)
    _, _, test_df = temporal_split(df_feat)
    test_splits[stype] = test_df
    print(f"{stype}: {len(test_df):,} test rows")
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. AUC-PR and AUC-ROC curves — eMBB 30min"))

cells.append(nbf.v4.new_code_cell("""\
stype, horizon = "eMBB", 30
clf, threshold = load_classifier(stype, horizon, models_dir=Path("../models"))
X_test, y_test = _prepare_X_y(test_splits[stype], horizon)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, ax=axes[0], name="XGBoost")
axes[0].axvline(x=recall_score(y_test, (clf.predict_proba(X_test)[:,1] >= threshold).astype(int)),
                color="red", linestyle="--", label=f"Selected threshold={threshold:.3f}")
axes[0].set_title(f"Precision-Recall Curve — {stype} {horizon}min")
axes[0].legend()

RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=axes[1], name="XGBoost")
axes[1].set_title(f"ROC Curve — {stype} {horizon}min")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Confusion matrices — all 9 models"))

cells.append(nbf.v4.new_code_cell("""\
horizons = [15, 30, 60]
slice_types = ["eMBB", "URLLC", "mMTC"]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, stype in enumerate(slice_types):
    for j, h in enumerate(horizons):
        clf, thr = load_classifier(stype, h, models_dir=Path("../models"))
        X_test, y_test = _prepare_X_y(test_splits[stype], h)
        y_pred = (clf.predict_proba(X_test)[:,1] >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=axes[i][j], colorbar=False)
        axes[i][j].set_title(f"{stype} {h}min  (thr={thr:.2f})")

plt.suptitle("Confusion Matrices — All 9 Models", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Metrics summary table"))

cells.append(nbf.v4.new_code_cell("""\
rows = []
for stype in slice_types:
    for h in horizons:
        clf, thr = load_classifier(stype, h, models_dir=Path("../models"))
        X_test, y_test = _prepare_X_y(test_splits[stype], h)
        y_proba = clf.predict_proba(X_test)[:,1]
        y_pred  = (y_proba >= thr).astype(int)
        rows.append({
            "Slice": stype, "Horizon": f"{h}min",
            "Threshold": round(thr, 3),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 3),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 3),
            "AUC-ROC":   round(roc_auc_score(y_test, y_proba), 3),
            "AUC-PR":    round(average_precision_score(y_test, y_proba), 3),
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Feature importance — top-20 XGBoost gain scores (eMBB 30min)"))

cells.append(nbf.v4.new_code_cell("""\
clf, _ = load_classifier("eMBB", 30, models_dir=Path("../models"))
X_test, _ = _prepare_X_y(test_splits["eMBB"], 30)

importances = clf.feature_importances_
feat_names  = X_test.columns.tolist()

idx = np.argsort(importances)[-20:][::-1]
top_feats  = [feat_names[i] for i in idx]
top_scores = importances[idx]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_feats[::-1], top_scores[::-1], color="#4C9BE8")
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("Top-20 Feature Importance — eMBB 30min Classifier")
plt.tight_layout()
plt.show()
"""))

nb.cells = cells

out = Path("notebooks/05_classifier.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {out}")
