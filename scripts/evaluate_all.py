"""
Master Evaluation & Audit Script
==================================
Evaluates the trained model using standard ML metrics from the syllabus:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
  - Confusion Matrix
  - Real-world human query benchmark
  - Data integrity checks

Uses: Python, NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Scikit-Learn Metrics (Module 4: Model Evaluation)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Visualization (Module 2)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")

for d in [DOCS_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

from utils.helpers import parse_symptoms_with_metadata


def run_master_evaluation():
    print("=" * 70)
    print("SAARTHI MEDICAL DIAGNOSIS AI - EVALUATION & AUDIT REPORT")
    print("=" * 70)

    audit = {}

    # ──────────────────────────────────────────────────────
    # 1. DATA INTEGRITY CHECK (Module 2: Data Preprocessing)
    # ──────────────────────────────────────────────────────
    print("\n[1/6] Data Integrity Check...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "Testing.csv"))

    symptom_cols = [c for c in train_df.columns if c != "prognosis"]
    X_train = train_df[symptom_cols]
    X_test = test_df[symptom_cols]

    # Check for duplicates and missing values
    train_dups = train_df.duplicated().sum()
    missing = X_train.isnull().sum().sum()
    leakage = pd.merge(X_train, X_test, on=symptom_cols, how="inner")

    print(f"  Training rows:      {len(train_df)}")
    print(f"  Testing rows:       {len(test_df)}")
    print(f"  Duplicates:         {train_dups}")
    print(f"  Missing values:     {missing}")
    print(f"  Train-Test leakage: {len(leakage)} rows")

    audit["data_integrity"] = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "duplicates": int(train_dups),
        "missing_values": int(missing),
        "leakage_rows": len(leakage),
        "status": "PASSED" if len(leakage) == 0 else "FAILED"
    }

    # ──────────────────────────────────────────────────────
    # 2. LOAD MODEL (Module 1: File Handling)
    # ──────────────────────────────────────────────────────
    print("\n[2/6] Loading Model Artifacts...")
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))

    best_name = "Best Model"
    name_path = os.path.join(MODELS_DIR, "best_model_name.pkl")
    if os.path.exists(name_path):
        best_name = joblib.load(name_path)

    for model_file in ["best_model.pkl", "naive_bayes.pkl", "random_forest.pkl"]:
        p = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(p):
            model = joblib.load(p)
            break

    print(f"  Active Model: {best_name}")

    # ──────────────────────────────────────────────────────
    # 3. TEST SET EVALUATION (Module 4: Model Evaluation)
    # ──────────────────────────────────────────────────────
    print("\n[3/6] Evaluating on Test Set (Standard ML Metrics)...")

    y_test = test_df["prognosis"]
    y_test_enc = encoder.transform(y_test)
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    # Standard Metrics (Module 4)
    acc = accuracy_score(y_test_enc, y_pred) * 100
    prec = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0) * 100
    rec = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0) * 100

    try:
        y_prob = model.predict_proba(X_test_scaled)
        roc = roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="weighted") * 100
    except Exception:
        roc = 0.0

    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1 Score:  {f1:.2f}%")
    print(f"  ROC-AUC:   {roc:.2f}%")

    audit["test_evaluation"] = {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1_score": round(f1, 2),
        "roc_auc": round(roc, 2)
    }

    # ──────────────────────────────────────────────────────
    # 4. HUMAN QUERY EVALUATION
    # ──────────────────────────────────────────────────────
    print("\n[4/6] Evaluating on Real-World Human Queries...")

    queries_path = os.path.join(DATA_DIR, "human_eval_queries.json")
    if os.path.exists(queries_path):
        with open(queries_path, "r") as f:
            human_queries = json.load(f)

        top1_correct = 0
        top3_correct = 0

        for item in human_queries:
            text = item["text"]
            expected = item["expected"]

            meta = parse_symptoms_with_metadata(text, symptom_columns)
            syms = meta["matched"]

            feat = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)
            for s in syms:
                if s in feat.columns:
                    feat[s] = 1

            scaled = scaler.transform(feat)
            probs = model.predict_proba(scaled)[0]

            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_diseases = encoder.inverse_transform(top3_idx)

            if top3_diseases[0].lower().strip() == expected.lower().strip():
                top1_correct += 1
            if any(d.lower().strip() == expected.lower().strip() for d in top3_diseases):
                top3_correct += 1

        total = len(human_queries)
        top1_acc = top1_correct / total * 100
        top3_acc = top3_correct / total * 100

        print(f"  Total queries:  {total}")
        print(f"  Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total})")
        print(f"  Top-3 Accuracy: {top3_acc:.2f}% ({top3_correct}/{total})")

        audit["human_evaluation"] = {
            "total": total,
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2)
        }
    else:
        print("  [SKIP] human_eval_queries.json not found")

    # ──────────────────────────────────────────────────────
    # 5. ROBUSTNESS CHECKS
    # ──────────────────────────────────────────────────────
    print("\n[5/6] Robustness Checks...")

    # Order invariance test
    seq1 = ["cough", "high_fever", "breathlessness", "chest_pain"]
    seq2 = ["chest_pain", "breathlessness", "high_fever", "cough"]

    f1_vec = scaler.transform(pd.DataFrame([[1 if c in seq1 else 0 for c in symptom_columns]], columns=symptom_columns))
    f2_vec = scaler.transform(pd.DataFrame([[1 if c in seq2 else 0 for c in symptom_columns]], columns=symptom_columns))
    p1 = model.predict(f1_vec)
    p2 = model.predict(f2_vec)
    order_pass = np.array_equal(p1, p2)

    # Overconfidence check
    mixed_syms = ["itching", "chest_pain", "vomiting", "joint_pain"]
    f_mixed = scaler.transform(pd.DataFrame([[1 if c in mixed_syms else 0 for c in symptom_columns]], columns=symptom_columns))
    max_prob = np.max(model.predict_proba(f_mixed)[0]) * 100
    overconf_pass = max_prob < 99.9

    print(f"  Order Invariance:      {'PASSED' if order_pass else 'FAILED'}")
    print(f"  Overconfidence Check:  {'PASSED' if overconf_pass else 'FAILED'} (Max: {max_prob:.1f}%)")

    audit["robustness"] = {
        "order_invariance": "PASSED" if order_pass else "FAILED",
        "overconfidence_check": "PASSED" if overconf_pass else "FAILED",
        "max_confidence": round(float(max_prob), 2)
    }

    # ──────────────────────────────────────────────────────
    # 6. SAVE REPORTS (Module 1: File Handling)
    # ──────────────────────────────────────────────────────
    print("\n[6/6] Saving Reports...")

    # JSON report
    json_path = os.path.join(DOCS_DIR, "ml_audit_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # Markdown report
    h_top1 = audit.get("human_evaluation", {}).get("top1_accuracy", "N/A")
    h_top3 = audit.get("human_evaluation", {}).get("top3_accuracy", "N/A")

    report = f"""# Saarthi Medical Diagnosis AI - Evaluation Report

**Model:** {best_name}
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## Data Integrity
| Check | Result |
|-------|--------|
| Training rows | {audit['data_integrity']['train_rows']} |
| Testing rows | {audit['data_integrity']['test_rows']} |
| Duplicates | {audit['data_integrity']['duplicates']} |
| Missing values | {audit['data_integrity']['missing_values']} |
| Train-Test leakage | {audit['data_integrity']['leakage_rows']} rows |

## Model Performance (Standard ML Metrics)
| Metric | Score |
|--------|-------|
| Accuracy | {acc:.2f}% |
| Precision | {prec:.2f}% |
| Recall | {rec:.2f}% |
| F1 Score | {f1:.2f}% |
| ROC-AUC | {roc:.2f}% |

## Human Query Evaluation
| Metric | Score |
|--------|-------|
| Top-1 Accuracy | {h_top1}% |
| Top-3 Accuracy | {h_top3}% |

## Robustness
| Test | Result |
|------|--------|
| Order Invariance | {audit['robustness']['order_invariance']} |
| Overconfidence Check | {audit['robustness']['overconfidence_check']} |

---
*Generated by scripts/evaluate_all.py*
"""
    md_path = os.path.join(DOCS_DIR, "Evaluation_Audit_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  [SAVED] {json_path}")
    print(f"  [SAVED] {md_path}")

    print("\n" + "=" * 70)
    print("[OK] Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_master_evaluation()
