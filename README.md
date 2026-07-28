# 🩺 Saarthi — AI-Powered Medical Diagnosis System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Saarthi** is an advanced clinical decision support platform that predicts probable medical diagnoses across **168 disease categories** using **397 symptom indicators** trained on **25,200 medical records**.

---

## 📌 Features

- **⚡ Instant Real-Time Inference:** Optimized serialization enables <15 ms prediction latency.
- **💬 Free-Text Conversational NLP Engine:**
  - Free-text symptoms parsing with English & **Hinglish** medical terminology.
  - Automatic fuzzy spelling correction (e.g. `fevr` → `high_fever`, `bukhar` → `high_fever`).
  - Matched vs. unmatched symptom metadata reporting.
- **🏥 Comprehensive Disease Knowledge Base:** Detailed disease cards including recommended specialists, risk factors, precautions, recovery times, and emergency warning signs.
- **🤖 Interactive ML Studio:** Performance comparison across 6 algorithms:
  1. **Calibrated Naive Bayes** (*Primary Active Model — 95.5% Top-2 Accuracy*)
  2. **Random Forest (Tuned)**
  3. **Extra Trees Classifier**
  4. **Logistic Regression**
  5. **Decision Tree Classifier**
  6. **Gaussian Naive Bayes**
- **📊 Interactive Data Analytics:** Visual dashboards for class balance, symptom frequency, and co-occurrence patterns.
- **📜 Patient Session History:** JSON-backed prediction logging with CSV export capabilities.

---

## 📁 Repository Structure

```
Medical-Diagnosis-AI/
│
├── app.py                           # Main Streamlit Entry Point
├── requirements.txt                 # Project Dependencies
├── README.md                        # Master Documentation
├── .gitignore                       # Git Exclusion Rules
│
├── assets/                          # Static Web Assets (Favicon, Custom CSS)
│   ├── favicon.png
│   └── style.css
│
├── data/                            # Dataset Pipeline & Schema
│   ├── Training.csv                 # Master Dataset (25,200 rows x 397 features)
│   ├── Testing.csv                  # Standard Benchmark Test Set
│   ├── disease_info.csv             # Reference Metadata (168 diseases)
│   ├── symptom_disease_dataset.csv  # Augmented Raw Dataset
│   ├── generate_disease_info.py     # Metadata Generation Script
│   └── prepare_dataset.py           # Core Preprocessing & Schema Builder
│
├── docs/                            # Project Reports & Benchmarks
│   ├── top2_optimization_report.md  # 95.5% Top-2 Accuracy Benchmark Report
│   └── qa_1000_test_report.md       # 1,000-Case QA Evaluation Suite
│
├── models/                          # OOP ML Modules & Production Artifacts
│   ├── __init__.py
│   ├── predictor.py                 # DiseasePredictor High-Level Class
│   ├── trainer.py                   # ModelTrainer Pipeline Class
│   ├── calibrated_nb.pkl            # Primary Production Classifier
│   ├── random_forest_tuned.pkl      # Tuned Random Forest Model
│   ├── encoder.pkl                  # Target Label Encoder
│   ├── scaler.pkl                   # Standard Feature Scaler
│   ├── symptom_columns.pkl          # Indexed Feature Schema (397 symptoms)
│   ├── best_model_name.pkl          # Active Model Selector Pointer
│   └── model_results.csv            # Studio Performance Benchmarks
│
├── notebooks/                       # Reconstructed Jupyter Notebooks
│   ├── 01_Data_Preparation_and_EDA.ipynb
│   ├── 02_Model_Training_and_Evaluation.ipynb
│   └── 03_NLP_Extraction_and_Prediction_Pipeline.ipynb
│
├── pages/                           # Streamlit Multi-Page Modules
│   ├── 1_🏥_Disease_Database.py     # Disease Knowledge Base
│   ├── 2_🤖_ML_Studio.py            # Model Benchmarking Studio
│   ├── 3_📊_Data_Analytics.py       # EDA Dashboards
│   ├── 4_📜_Prediction_History.py   # Patient Session Logs
│   └── 5_ℹ️_About_Saarthi.py        # Architecture Metadata
│
├── preprocessing/                   # Data Transformation Pipelines
│   ├── __init__.py
│   ├── cleaner.py                   # Symptom Normalization Engine
│   ├── encoder.py                   # Categorical Encoding Wrappers
│   └── scaler.py                    # Feature Scaling Utilities
│
├── scripts/                         # Reproducible Pipelines & Tests
│   ├── train_all.py                 # Multi-Model Training Script
│   ├── train_optimized_pipeline.py  # Ensembling & Calibration Trainer
│   └── test_1000_qa_suite.py        # 1,000-Case Automated Test Suite
│
├── utils/                           # Data Loaders & Helpers
│   ├── __init__.py
│   ├── data_loader.py               # Streamlit Cached Data Loaders
│   ├── helpers.py                   # NLP Engine & Symptom Parser
│   └── theme.py                     # UI Theme Components
│
└── visualizations/                  # Chart Plotting Components
    ├── __init__.py
    └── charts.py                    # Plotly & Seaborn Dashboards
```

---

## 🛠️ Technology Stack

- **Python 3.12** — Primary Development Language
- **Streamlit** — Web Application Framework
- **Scikit-Learn** — Machine Learning Models & Scaling Pipelines
- **Pandas & NumPy** — Data Manipulation & Numeric Computation
- **Plotly & Seaborn** — Interactive Data Visualizations
- **Joblib** — Model Serialization

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/jaidevxr/saarthi-medical-diagnosis.git
cd saarthi-medical-diagnosis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Evaluation Results

| Metric | Performance Score |
|---|---|
| **Top-1 Accuracy** | **88.70%** |
| **Top-2 Accuracy** | **95.50%** |
| **Top-3 Accuracy** | **96.50%** |
| **Top-5 Accuracy** | **97.50%** |
| **Macro F1-Score** | **0.8890** |
| **Avg Inference Time** | **12.40 ms** |
| **Robustness Score** | **100% (32 / 32 Edge Cases)** |

---

## 🔬 Future Scope

1. **Integration with EHR / EMR Systems:** Exporting prediction logs in HL7/FHIR standards.
2. **Multilingual Speech Input:** Voice-to-text symptom input in regional Indian languages.
3. **Lab Parameter Fusion:** Combining symptom profiles with blood test and biomarker values.

---

## ⚠️ Medical Disclaimer

Saarthi is designed strictly as an **educational clinical decision support system**. It does **NOT** provide medical advice, diagnosis, or treatment. Always consult a qualified medical professional for health concerns.
