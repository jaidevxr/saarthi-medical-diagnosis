"""
Pre-train all 6 Machine Learning models on the Master Medical Dataset (20,850 rows x 376 symptoms).
"""

import os, sys, time
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_training_data
from preprocessing.encoder import DataEncoder
from preprocessing.scaler import DataScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_all():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading training dataset...")
    df = load_training_data()
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    X = df[symptom_cols]
    y = df["prognosis"]

    print(f"Dataset shape: {df.shape} ({len(symptom_cols)} symptoms, {y.nunique()} diseases)")

    encoder = DataEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = DataScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(symptom_cols, os.path.join(MODELS_DIR, "symptom_columns.pkl"))

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_test)
        t_pred = time.time() - t1

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Train time: {t_train:.2f}s")

        results.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Training Time (s)": round(t_train, 3),
            "Prediction Time (s)": round(t_pred, 4),
        })

        model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)

    best_idx = results_df["Accuracy"].idxmax()
    best_name = results_df.loc[best_idx, "Algorithm"]
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
    print(f"\nAll models saved. Best model: {best_name} ({results_df.loc[best_idx, 'Accuracy']:.2%})")

if __name__ == "__main__":
    train_all()
