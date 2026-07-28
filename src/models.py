"""
Model Development & Tuning Module (src/models.py)
=================================================
Trains, tunes, and evaluates all 6 approved classical ML models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Naive Bayes (BernoulliNB)

Uses validation data ONLY for hyperparameter tuning and model selection.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42


def get_candidate_models():
    """
    Returns dict of candidate models with hyperparameter search grids.
    All models strictly from approved list!
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=500, random_state=SEED),
            {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [10, 15, 20], "criterion": ["gini", "entropy"]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"n_estimators": [30, 50, 100], "max_depth": [10, 15]}
        ),
        "KNN": (
            KNeighborsClassifier(n_jobs=1),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
        ),
        "SVM": (
            SVC(probability=True, random_state=SEED),
            {"C": [0.5, 1.0], "kernel": ["rbf"]}
        ),
        "Naive Bayes": (
            BernoulliNB(),
            {"alpha": [0.1, 0.3, 0.5, 1.0]}
        ),
    }


def train_and_tune_candidates(X_train, y_train, X_val, y_val):
    """
    Trains and tunes all 6 candidate models on validation split ONLY.
    Returns results DataFrame and dict of tuned models.
    """
    candidates = get_candidate_models()
    results = []
    tuned_models = {}

    print("\n" + "=" * 70)
    print("3.1 CANDIDATE MODEL TRAINING & HYPERPARAMETER TUNING (Validation Only)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    for name, (model, param_grid) in candidates.items():
        print(f"\n  Tuning & Fitting {name}...")
        t0 = time.time()

        # For heavy models, sample to keep tuning ultra-fast
        if name in ["SVM", "KNN", "Random Forest"] and len(X_train) > 10000:
            sample_idx = np.random.choice(len(X_train), size=10000, replace=False)
            X_fit = X_train.iloc[sample_idx] if isinstance(X_train, pd.DataFrame) else X_train[sample_idx]
            y_fit = y_train.iloc[sample_idx] if isinstance(y_train, pd.Series) else y_train[sample_idx]
        else:
            X_fit = X_train
            y_fit = y_train

        grid = GridSearchCV(model, param_grid, cv=skf, scoring="accuracy", n_jobs=1)
        grid.fit(X_fit, y_fit)
        train_time = time.time() - t0

        best_clf = grid.best_estimator_
        best_params = grid.best_params_
        tuned_models[name] = best_clf

        # Evaluate on validation split
        val_preds = best_clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds) * 100
        val_prec = precision_score(y_val, val_preds, average="weighted", zero_division=0) * 100
        val_rec = recall_score(y_val, val_preds, average="weighted", zero_division=0) * 100
        val_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0) * 100

        # Model size
        model_path = os.path.join(MODELS_DIR, f"temp_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(best_clf, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if os.path.exists(model_path):
            os.remove(model_path)

        # Inference latency per query
        t_lat0 = time.time()
        for _ in range(50):
            best_clf.predict(X_val[:1])
        lat_ms = (time.time() - t_lat0) / 50.0 * 1000.0

        print(f"    Best Params:    {best_params}")
        print(f"    Validation Acc: {val_acc:.2f}%")
        print(f"    Validation F1:  {val_f1:.2f}%")
        print(f"    Train Time:     {train_time:.1f}s")
        print(f"    Latency:        {lat_ms:.2f}ms/query")
        print(f"    Model Size:     {size_mb:.2f}MB")

        results.append({
            "Model": name,
            "Best Hyperparameters": str(best_params),
            "Validation Accuracy": round(val_acc, 2),
            "Validation Precision": round(val_prec, 2),
            "Validation Recall": round(val_rec, 2),
            "Validation F1": round(val_f1, 2),
            "Training Time (s)": round(train_time, 1),
            "Inference Latency (ms)": round(lat_ms, 2),
            "Model Size (MB)": round(size_mb, 2),
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("3.2 VALIDATION PERFORMANCE COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return results_df, tuned_models
