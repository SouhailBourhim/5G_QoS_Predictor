# 5G QoS Predictor: Machine Learning Project Introduction

---

## Frame the Problem and Look at the Big Picture

### 1. Define the objective in business terms

**Business Objective:**
Reduce SLA violation-related revenue loss and customer churn in 5G network slicing operations by predicting SLA breaches 15–60 minutes before they occur, enabling proactive resource reallocation and congestion mitigation.

**Quantified Impact:**

* **Current state:** Mobile network operators face financial penalties (€10,000–€100,000 per incident for enterprise URLLC contracts), service degradation costs, and customer churn when SLA violations occur.
* **Target state:** Achieve 90%+ detection rate of upcoming violations with 30-minute advance warning, allowing operations teams to prevent 70%+ of actual breaches through proactive intervention.
* **Business value:** For a mid-sized operator with 50 enterprise slice customers, preventing even 10 SLA violations per month could save €500,000–€1M annually in penalties, plus preserved revenue from retained customers.

---

### 2. How will your solution be used?

**Primary Users:**
Network Operations Center (NOC) analysts, automated SON orchestrators, and enterprise customer dashboards.

**Usage Workflow:**

```text
REAL-TIME MONITORING → RISK PREDICTION → DECISION → MITIGATION → CONTINUOUS LEARNING
```

**Integration Points:**

* **Input:** OSS/BSS systems, NWDAF data collection, RAN/Core KPI exporters
* **Output:** Alarms to NOC dashboards, API calls to orchestrators (ONAP, OSM), incident tickets

---

### 3. What are the current solutions/workarounds?

**Current Industry Practice: Static Threshold Alarms**

```python
if latency > SLA_THRESHOLD * 0.95:
    trigger_alarm()
```

**Limitations:**

* Reactive, not predictive
* No temporal context
* High false alarm rate
* No cross-KPI intelligence
* Manual root cause analysis

**Workarounds:**

* Manual monitoring by NOC analysts
* Over-provisioning resources
* Conservative SLA thresholds

---

### 4. How should you frame this problem?

**Primary Task:** Supervised Binary Classification (Multi-Horizon)

```text
Given: X(t)
Predict: Y_h(t) ∈ {0,1}, h ∈ {15,30,60}
```

**Secondary Task:** Time-Series Regression

```text
Predict future KPI values V_k(t+h)
```

**Training:** Offline batch retraining
**Inference:** Online real-time prediction every 5 minutes

---

### 5. How should performance be measured?

**Primary Metric:** F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Primary Targets:**

* Recall ≥ 90%
* Precision ≥ 60%
* F1 ≥ 0.75

**Secondary Metrics:**

* Precision-Recall AUC
* Median lead time
* Per-slice recall
* Per-event-type recall

---

### 6. Is performance aligned with business objectives?

| Business Goal          | Technical Metric | Threshold |
| ---------------------- | ---------------- | --------- |
| Prevent 70% violations | Recall           | ≥ 90%     |
| Minimize alert fatigue | Precision        | ≥ 60%     |
| Actionable warning     | Lead time        | ≥ 25 min  |

---

### 7. Minimum viable performance

* Recall ≥ 75%
* Precision ≥ 50%
* F1 ≥ 0.60
* Lead time ≥ 15 min
* Better than static threshold baseline

---

### 8. Comparable problems

* Predictive maintenance
* ICU deterioration prediction
* Financial crash forecasting
* NetSentinel anomaly detection

---

### 9. Human expertise available

**Telecom Experts:**

* INPT faculty
* Industry engineers
* 3GPP references

**ML Experts:**

* Faculty
* Senior students
* Online ML communities

---

### 10. Manual solution workflow

```text
Monitor KPIs → Detect trends → Estimate breach time → Mitigate
```

**Features implied by human workflow:**

* Raw KPI values
* Lag features
* Rate of change
* SLA margin
* Time-of-day encoding
* Cross-slice resource features

---

### 11. Assumptions

**Data Assumptions:**

* 5-minute granularity is sufficient
* Synthetic scenarios reflect reality
* 12-hour lookback window is enough

**Model Assumptions:**

* Feature sufficiency
* Good generalization
* Stable traffic patterns

**Business Assumptions:**

* Operators trust alerts
* 15–60 min lead time is actionable

---

### 12. Assumption verification

**During EDA:**

* KPI autocorrelation
* Event realism
* Temporal coverage

**Post Deployment:**

* Drift monitoring
* Operator response time
* Alert fatigue

---

## Summary Table

| Aspect      | Definition                          |
| ----------- | ----------------------------------- |
| Objective   | Predict SLA violations in advance   |
| Users       | NOC + orchestrators                 |
| ML Task     | Binary classification + forecasting |
| Main Metric | F1 / Recall                         |
| Success     | 2× static baseline                  |

---

## Next Steps: Get the Data

1. Download 5G-NIDD
2. Perform EDA
3. Build synthetic data generator
4. Generate 90-day dataset
5. Train XGBoost / LSTM
6. Evaluate lead time

---

## Project Checklist

* [x] Problem framing
* [x] Objective definition
* [x] Metrics
* [x] Assumptions
* [ ] Data acquisition
* [ ] EDA
* [ ] Feature engineering
* [ ] Modeling
* [ ] Evaluation
* [ ] Deployment

---

**Document Version:** 1.0
**Last Updated:** April 2026
**Author:** Souhail Bourhim
**Project:** 5G QoS Predictor — Network Slice SLA Violation Forecasting
