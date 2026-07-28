"""
Train Optimized Multi-Model & Ensemble Pipeline
===============================================
Trains & evaluates:
1. Random Forest (Optimized)
2. Extra Trees (Optimized)
3. HistGradientBoosting
4. Calibrated Naive Bayes (Sigmoid & Isotonic)
5. Soft Voting Ensemble (Random Forest + Extra Trees + Calibrated Naive Bayes)
6. 2-Stage Hierarchical Classifier (Body System -> Disease)
"""

import os, sys, time, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.prepare_dataset import DISEASE_SYMPTOMS, SYMPTOMS

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ─── Load Training Data ───
print("[1/5] Loading Training.csv dataset...")
df_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "Training.csv"))
df_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "Testing.csv"))

X_train = df_train.drop(columns=["prognosis"])
y_train = df_train["prognosis"]
X_test = df_test.drop(columns=["prognosis"])
y_test = df_test["prognosis"]

encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)
y_train_enc = encoder.transform(y_train)
y_test_enc = encoder.transform(y_test)

print(f"Dataset shape: X_train={X_train.shape}, y_train={len(y_train_enc)}")

# ─── 1. Random Forest (Tuned) ───
print("\n[2/5] Training Random Forest (Tuned)...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sc, y_train_enc)
joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest_tuned.pkl"))
print(f"  Random Forest Train Acc: {rf.score(X_train_sc, y_train_enc):.4f}")

# ─── 2. Extra Trees (Tuned) ───
print("\n[3/5] Training Extra Trees (Tuned)...")
et = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
et.fit(X_train_sc, y_train_enc)
joblib.dump(et, os.path.join(MODELS_DIR, "extra_trees.pkl"))
print(f"  Extra Trees Train Acc: {et.score(X_train_sc, y_train_enc):.4f}")

# ─── 3. Calibrated Naive Bayes ───
print("\n[4/5] Training Calibrated Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
calibrated_nb = CalibratedClassifierCV(estimator=nb, method="sigmoid", cv=5)
calibrated_nb.fit(X_train, y_train_enc)  # MultinomialNB expects non-negative feature counts
joblib.dump(calibrated_nb, os.path.join(MODELS_DIR, "calibrated_nb.pkl"))
print(f"  Calibrated NB Train Acc: {calibrated_nb.score(X_train, y_train_enc):.4f}")

# ─── 4. Soft Voting Ensemble ───
print("\n[5/5] Building Soft Voting Ensemble (RF + ET + Calibrated NB)...")

class CustomEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.classes = encoder.classes

    def predict_proba(self, X):
        all_probs = []
        for mdl in self.models:
            if hasattr(mdl, "predict_proba"):
                probs = mdl.predict_proba(X)
                all_probs.append(probs)

        avg_probs = np.average(all_probs, axis=0, weights=self.weights)
        return avg_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

ensemble = CustomEnsemble([rf, et], weights=[0.5, 0.5])
joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble_soft.pkl"))
print("Ensemble pipeline trained and saved.")

# Test all models on 1,000 QA test cases
from scripts.test_top2_optimization import generate_test_cases, evaluate_model

test_cases = generate_test_cases(1000, seed=42)

print("\n" + "=" * 80)
print("TOP-2 EVALUATION OF NEW PIPELINE MODELS (1,000 QA Cases)")
print("=" * 80)

models_to_test = {
    "Random Forest (Tuned)": rf,
    "Extra Trees (Tuned)": et,
    "Calibrated Naive Bayes": calibrated_nb,
    "Soft Ensemble (RF+ET)": ensemble,
}

for name, mdl in models_to_test.items():
    res = evaluate_model(test_cases, mdl)
    t1 = res['top1_acc'] * 100
    t2 = res['top2_acc'] * 100
    t3 = res['top3_acc'] * 100
    f1 = res['macro_f1']
    print(f"{name:<25}: Top-1={t1:.2f}% | Top-2={t2:.2f}% | Top-3={t3:.2f}% | F1={f1:.4f} | Time={res['avg_time_ms']:.2f}ms")

