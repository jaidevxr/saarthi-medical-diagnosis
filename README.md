# 🩺 Saarthi — Master Medical Diagnosis Prediction System

**Saarthi** is an advanced, production-quality Machine Learning-powered clinical decision support platform. It predicts probable disease diagnoses across **139 diseases** covering **12 medical categories** using **376 symptom indicators** trained on **20,850 medical records**.

---

## 🌟 Key Features

- **⚡ Instant Real-Time Prediction**: Zero load times with cached ML models pre-serialized for fast predictions.
- **💬 Dual Symptom Input Engine**:
  - **Natural Language Description (Default)**: Type or paste how you feel in free text (e.g. *"I have high fever, chills, severe headache, joint pain, and nausea"*). Real-time NLP automatically detects symptoms and computes differential diagnoses instantly.
  - **Structured Symptom Search**: Search and select from 376 indexed symptom markers.
- **🏥 Master Knowledge Base**: 139 disease profiles complete with recommended medical specialists, severity ratings, common causes, risk factors, precautions, dietary advice, and emergency warning signs.
- **🤖 Machine Learning Studio**: Benchmark and switch between 6 classical ML classification algorithms:
  1. **Random Forest Classifier** (*98.75% Accuracy — Active*)
  2. **Logistic Regression** (*98.32% Accuracy*)
  3. **Support Vector Machine (SVM)** (*97.39% Accuracy*)
  4. **K-Nearest Neighbors (KNN)** (*89.14% Accuracy*)
  5. **Decision Tree Classifier** (*85.44% Accuracy*)
  6. **Naive Bayes (Gaussian)** (*57.36% Accuracy*)
- **📊 Data Analytics & EDA**: Interactive visual dashboards of symptom prevalence, class balance, and correlation heatmaps.
- **📜 Patient Session History**: Searchable prediction logs with CSV export.

---

## 🩺 Medical Specialties Covered (139 Diseases)

1. **🫁 Respiratory Diseases** (Asthma, Pneumonia, COPD, Bronchitis, Tuberculosis, Sinusitis, etc.)
2. **🫀 Cardiovascular Disorders** (Heart Attack, Hypertension, Angina, DVT, Pericarditis, etc.)
3. **🧠 Neurological Conditions** (Migraine, Epilepsy, Meningitis, Bell's Palsy, Sciatica, Stroke, etc.)
4. **🧴 Skin & Dermatology** (Eczema, Psoriasis, Acne, Cellulitis, Shingles, Rosacea, Scabies, etc.)
5. **🪵 Gastrointestinal Diseases** (GERD, Peptic Ulcer, Hepatitis A-E, IBS, Appendicitis, Pancreatitis, etc.)
6. **🧫 Kidney & Urinary System** (UTI, Kidney Stones, CKD, Pyelonephritis, Prostatitis, etc.)
7. **🩸 Endocrine & Metabolic** (Diabetes, Hypothyroidism, Hyperthyroidism, PCOS, Gout, Cushing's, etc.)
8. **🦠 Infectious Diseases** (Malaria, Dengue, Typhoid, Chickenpox, Measles, Lyme, Cholera, etc.)
9. **👂 ENT (Ear, Nose, Throat)** (Otitis Media, Tonsillitis, Meniere's, Vertigo, Laryngitis, etc.)
10. **👁 Ophthalmology (Eye)** (Conjunctivitis, Glaucoma, Uveitis, Dry Eye Syndrome, Blepharitis, etc.)
11. **🦴 Rheumatology & Autoimmune** (Osteoarthritis, Rheumatoid Arthritis, Lupus, MS, Fibromyalgia, etc.)
12. **🩺 General & Emergency Medicine** (Anemia, Heat Stroke, Dehydration, Chronic Fatigue, etc.)

---

## 🛠️ Technology Stack

- **Python 3.12** — Core Development Language
- **Streamlit** — Web Application Framework
- **Scikit-Learn** — Machine Learning Classifiers & Preprocessing Pipeline
- **Pandas & NumPy** — High-Performance Data Processing
- **Plotly & Seaborn** — Interactive Data Visualizations
- **Joblib** — High-Performance Model Serialization

---

## 🚀 Quick Start Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/saarthi-medical-diagnosis.git
cd saarthi-medical-diagnosis

pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 👥 Contributors

- **Khushi** — Lead Data Analysis & Medical Domain Mapping
- **Jai** — Machine Learning Architecture & Application Engineering

*(Feel free to contribute to Saarthi by submitting a pull request!)*

---

## ⚠️ Medical Disclaimer

Saarthi is designed as an **educational decision support system** powered by statistical machine learning. It is **NOT** a replacement for professional clinical diagnosis, advice, or treatment. Always consult a qualified medical professional for health concerns.
