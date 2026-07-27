"""
Build FULL ACCURACY (98.75%) Random Forest model for Saarthi (~12MB compressed).
"""

import os, sys
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_training_data
from preprocessing.encoder import DataEncoder
from preprocessing.scaler import DataScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

print("Training FULL ACCURACY Random Forest (n_estimators=100, full depth)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y_encoded)
acc = rf.score(X_scaled, y_encoded)
print(f"Random Forest Training Accuracy: {acc:.4%}")

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_scaled, y_encoded)

# Save artifacts with compression
joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"), compress=3)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"), compress=3)
joblib.dump(symptom_cols, os.path.join(MODELS_DIR, "symptom_columns.pkl"), compress=3)
joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"), compress=3)
joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"), compress=3)
joblib.dump("Random Forest", os.path.join(MODELS_DIR, "best_model_name.pkl"))

print("\nSaved Model Artifact Sizes:")
for f in os.listdir(MODELS_DIR):
    fp = os.path.join(MODELS_DIR, f)
    sz_mb = os.path.getsize(fp) / (1024 * 1024)
    print(f"  {f}: {sz_mb:.2f} MB")
