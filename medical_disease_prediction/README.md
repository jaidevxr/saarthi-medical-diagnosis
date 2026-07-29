# Medical Disease Prediction System

A lightweight, clean Machine Learning project for predicting medical diagnoses based on patient symptom profiles using classical classification algorithms.

## 🎯 Model Training & Selection

The notebook `data_cleaning_and_modeling.ipynb` trains and benchmarks 6 classical models:
1. **Logistic Regression** (Selected Best Model 🏆 - 97.65% Test Accuracy)
2. **Decision Tree**
3. **Random Forest**
4. **KNN (K-Nearest Neighbors)**
5. **Support Vector Machine (SVM)**
6. **Naive Bayes**

---

## 📂 Project Layout

```
medical_disease_prediction/
├── data_cleaning_and_modeling.ipynb  # Data cleaning, EDA & model training
├── app.py                            # Streamlit web UI
├── README.md
├── .gitignore
├── data/
│   ├── Training.csv
│   └── Testing.csv
└── saved_models/
    ├── best_model.joblib
    ├── label_encoder.joblib
    ├── feature_names.joblib
    ├── model_comparison.json
    └── symptoms.json
```

---

## 🚀 How to Run Streamlit Web App

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
streamlit run app.py
```
