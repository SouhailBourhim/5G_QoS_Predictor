# 5G QoS Predictor

> Predicting SLA violations in 5G network slices before they happen — An NWDAF Analytics Module Prototype

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project builds an end-to-end machine learning system that predicts Service Level Agreement (SLA) violations in 5G network slices (eMBB, URLLC, mMTC) **15 to 60 minutes before they occur**, enabling proactive resource management.

### Business Value
- **Problem:** Operators lose €50K-€100K per enterprise SLA violation
- **Solution:** ML-based early warning with 90%+ recall
- **Impact:** Prevent 70% of violations, estimated 775% annual ROI

### Technical Approach
- 🔬 **Domain-calibrated synthetic data** from real 5G measurements (5G-NIDD)
- 📊 **250+ telecom-informed features** (SLA margins, cross-slice competition, time-to-breach)
- 🤖 **Multi-horizon prediction** (15/30/60 min) with XGBoost, LightGBM, LSTM
- 📈 **Rigorous evaluation** (temporal CV, per-event-type analysis, SHAP explainability)
- 🚀 **Production deployment** (FastAPI + Streamlit + Docker)

## 🏗️ Project Structure

```text
5g-qos-predictor/
├── data/ # Raw, processed, and split datasets
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code modules
│ ├── data/ # Data generation and preprocessing
│ ├── features/ # Feature engineering
│ ├── models/ # ML models
│ └── evaluation/ # Evaluation framework
├── api/ # FastAPI REST API
├── dashboard/ # Streamlit dashboard
├── docker/ # Docker configuration
├── models/ # Trained model artifacts
├── tests/ # Unit tests
└── docs/ # Documentation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/5g-qos-predictor.git
cd 5g-qos-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
make data
# Or manually:
# python -m src.data.generator
```

### 3. Run Feature Engineering

```bash
make features
```

### 4. Train Models

```bash
make train
# Or manually:
# python -m src.models.classifier
# python -m src.models.forecaster
```

### 5. Launch Dashboard

```bash
make dashboard
# Access at http://localhost:8501
```

## 📊 Key Results

| Metric | Static Threshold | XGBoost (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| F₁ Score (30-min) | 0.34 | 0.84 | +147% |
| Recall | 0.42 | 0.91 | +117% |
| Median Lead Time | N/A | 28 min | — |

## 📚 Documentation

- Problem Framing
- Data Strategy
- Model Documentation

## 🔗 Related Projects

- **NetSentinel** — Network anomaly detection on IP flows (prior project)

## 👤 Author

**Souhail Bourhim**  
Engineering Student, INPT (Smart-ICT), Morocco

## 📄 License

This project is licensed under the MIT License.