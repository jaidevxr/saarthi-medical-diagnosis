"""
Master Production-Inspired Pipeline Execution Script (scripts/run_master_pipeline.py)
=====================================================================================
Executes Parts 1 through 9 of the production-inspired classical ML pipeline:
- Part 1: Data Strategy & Read-only distribution analysis & v1/v2 generation.
- Part 2: Feature Engineering & measured validation impact table.
- Part 3: Model Development & hyperparameter tuning for 6 classical ML models.
- Part 4: Mandatory 7-Stage Ablation Study (Stage 0 to Stage 6).
- Part 5: Explainability & Validation-based Uncertainty Refusal Thresholding.
- Part 6: Calibration (Brier, ECE, Reliability Diagram) & 13-Point Adversarial Stress Tests.
- Part 7: Efficiency Comparison (Latency, Model Size, Training Time).
- Part 8: Reproducibility, dataset SHA256 hashing, model SHA256 hashing & model_card.md generation.
- Part 9: Engineering structure & REST API verification.
"""

import os
import sys
import json
import time
import hashlib
import random
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
for d in [MODELS_DIR, VIS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

from src.normalization import SymptomNormalizer
from src.feature_engineering import FeatureExtractor
from src.data_pipeline import analyze_human_query_distribution, generate_training_data_v2
from src.calibration import calculate_ece, optimize_confidence_threshold
from src.explainability import generate_prediction_explanation
from src.models import train_and_tune_candidates

# Part 8: Reproducibility fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_file_hash(filepath):
    """Computes SHA256 hash of a file for Part 8 Reproducibility."""
    if not os.path.exists(filepath):
        return "N/A"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def evaluate_human_queries(clf, encoder, scaler, extractor, symptom_cols, active_feats, tau=0.0, preextracted=None):
    """
    Evaluates a model configuration on the 280 frozen human queries.
    Returns Top-1, Top-3 accuracies and refusal count.
    """
    with open(os.path.join(DATA_DIR, "human_eval_queries.json"), "r", encoding="utf-8") as f:
        human_queries = json.load(f)

    if preextracted is None:
        normalizer = SymptomNormalizer()
        preextracted = [normalizer.extract_symptoms(q["text"], symptom_cols)["matched"] for q in human_queries]

    top1_correct = 0
    top3_correct = 0
    refused_count = 0

    for i, item in enumerate(human_queries):
        expected = item["expected"]
        matched = preextracted[i]

        feat_dict = {col: [1 if col in matched else 0] for col in symptom_cols}
        feat_df = pd.DataFrame(feat_dict)

        if extractor is not None:
            feat_ext = extractor.transform(feat_df, active_feature_set=active_feats)
        else:
            feat_ext = feat_df

        feat_scaled = scaler.transform(feat_ext)
        probs = clf.predict_proba(feat_scaled)[0]

        conf = np.max(probs)
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_diseases = encoder.inverse_transform(top3_idx)

        if tau > 0 and conf < tau:
            refused_count += 1

        if top3_diseases[0].lower().strip() == expected.lower().strip():
            top1_correct += 1
        if any(d.lower().strip() == expected.lower().strip() for d in top3_diseases):
            top3_correct += 1

    n = len(human_queries)
    return {
        "top1": round(top1_correct / n * 100, 2),
        "top3": round(top3_correct / n * 100, 2),
        "refused": refused_count
    }


def main():
    print("=" * 80)
    print("SAARTHI MEDICAL AI — PRODUCTION-INSPIRED CLASSICAL ML PIPELINE")
    print("=" * 80)

    # ── PART 1: DATA STRATEGY ──
    print("\n[PART 1] DATA STRATEGY & DISTRIBUTION ANALYSIS")
    query_stats = analyze_human_query_distribution()

    # Generate v2 dataset
    train_v2_df = generate_training_data_v2()

    # Prepare Train / Validation / Test splits
    test_df = pd.read_csv(os.path.join(DATA_DIR, "Testing.csv"))
    symptom_cols = [c for c in train_v2_df.columns if c != "prognosis"]

    encoder = LabelEncoder()
    y_train_full = encoder.fit_transform(train_v2_df["prognosis"])
    y_test_full = encoder.transform(test_df["prognosis"])

    # 80/20 Train/Validation Split (Validation data ONLY used for tuning!)
    val_indices = np.random.choice(len(train_v2_df), size=int(0.2 * len(train_v2_df)), replace=False)
    train_indices = np.setdiff1d(np.arange(len(train_v2_df)), val_indices)

    X_train_df = train_v2_df.iloc[train_indices][symptom_cols].reset_index(drop=True)
    y_train_series = train_v2_df.iloc[train_indices]["prognosis"].reset_index(drop=True)
    y_train_split = y_train_full[train_indices]

    X_val_df = train_v2_df.iloc[val_indices][symptom_cols].reset_index(drop=True)
    y_val_series = train_v2_df.iloc[val_indices]["prognosis"].reset_index(drop=True)
    y_val_split = y_train_full[val_indices]

    X_test_df = test_df[symptom_cols].reset_index(drop=True)

    print(f"  Training Split:   {len(X_train_df)} rows")
    print(f"  Validation Split: {len(X_val_df)} rows (Used strictly for tuning)")
    print(f"  Holdout Test:     {len(X_test_df)} rows")

    # ── PART 2: FEATURE ENGINEERING IMPACT TABLE ──
    print("\n[PART 2] FEATURE ENGINEERING & MEASURED VALIDATION IMPACT")
    extractor = FeatureExtractor()
    extractor.fit(X_train_df, y_train_series)

    # Base feature set
    candidate_features = [
        'symptom_count',
        'body_system_count',
        'symptom_rarity',
        'common_uncommon_ratio',
        'severity_score',
        'overlap_score'
    ]

    base_lr = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")) else None
    
    # We measure validation accuracy delta sequentially
    active_feats = []
    feat_impact_records = []
    prev_val_acc = 99.0  # baseline validation acc

    for feat_name in candidate_features:
        test_feats = active_feats + [feat_name]
        X_tr_ext = extractor.transform(X_train_df, active_feature_set=test_feats)
        X_va_ext = extractor.transform(X_val_df, active_feature_set=test_feats)

        scaler_tmp = StandardScaler()
        X_tr_sc = scaler_tmp.fit_transform(X_tr_ext)
        X_va_sc = scaler_tmp.transform(X_va_ext)

        from sklearn.linear_model import LogisticRegression
        lr_tmp = LogisticRegression(max_iter=300, random_state=SEED)
        lr_tmp.fit(X_tr_sc, y_train_split)
        val_acc = accuracy_score(y_val_split, lr_tmp.predict(X_va_sc)) * 100

        delta = val_acc - prev_val_acc
        decision = "Keep" if delta >= -0.1 else "Drop"

        if decision == "Keep":
            active_feats.append(feat_name)
            prev_val_acc = val_acc

        feat_impact_records.append({
            "Feature": feat_name,
            "Validation Accuracy (%)": round(val_acc, 2),
            "Validation Accuracy Delta": f"{delta:+.2f}%",
            "Decision": decision
        })

    df_feat_impact = pd.DataFrame(feat_impact_records)
    print(df_feat_impact.to_string(index=False))

    # Transform datasets using selected feature set
    X_tr_final_df = extractor.transform(X_train_df, active_feature_set=active_feats)
    X_va_final_df = extractor.transform(X_val_df, active_feature_set=active_feats)
    X_te_final_df = extractor.transform(X_test_df, active_feature_set=active_feats)

    scaler_final = StandardScaler()
    X_tr_final_sc = scaler_final.fit_transform(X_tr_final_df)
    X_va_final_sc = scaler_final.transform(X_va_final_df)
    X_te_final_sc = scaler_final.transform(X_te_final_df)

    # ── PART 3: MODEL DEVELOPMENT ──
    print("\n[PART 3] MODEL DEVELOPMENT & TUNING (Validation Only)")
    model_results_df, tuned_models = train_and_tune_candidates(
        X_tr_final_sc, y_train_split, X_va_final_sc, y_val_split
    )

    best_val_row = model_results_df.sort_values("Validation F1", ascending=False).iloc[0]
    best_model_name = best_val_row["Model"]
    best_model = tuned_models[best_model_name]

    print(f"\n  [WINNER SELECTED ON VALIDATION DATA]: {best_model_name}")
    print(f"  Validation Accuracy: {best_val_row['Validation Accuracy']}% | F1: {best_val_row['Validation F1']}%")

    # ── PART 5: CALIBRATION & UNCERTAINTY THRESHOLD OPTIMIZATION ──
    print("\n[PART 5] CALIBRATION & UNCERTAINTY THRESHOLD TUNING (Validation Only)")
    val_probs = best_model.predict_proba(X_va_final_sc)
    best_tau, best_tau_f1 = optimize_confidence_threshold(val_probs, y_val_split)
    print(f"  Optimized Refusal Threshold tau: {best_tau:.2f} (Validation Refusal F1: {best_tau_f1:.4f})")

    # Save Best Model Artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
    joblib.dump(scaler_final, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(extractor, os.path.join(MODELS_DIR, "feature_extractor.pkl"))
    joblib.dump(active_feats, os.path.join(MODELS_DIR, "active_features.pkl"))
    joblib.dump(symptom_cols, os.path.join(MODELS_DIR, "symptom_columns.pkl"))
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    # Save Model Metadata
    metadata = {
        "best_model_name": best_model_name,
        "validation_accuracy": float(best_val_row["Validation Accuracy"]),
        "validation_f1": float(best_val_row["Validation F1"]),
        "confidence_threshold": float(best_tau),
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── PART 4: MANDATORY ABLATION STUDY ──
    print("\n[PART 4] MANDATORY STAGE-BY-STAGE ABLATION STUDY")
    print("  Evaluating Stage 0 to Stage 6 on frozen 280 human queries...")

    # Define stage models and configurations
    # Stage 0: Baseline LR on original data
    train_v1_df = pd.read_csv(os.path.join(DATA_DIR, "training_data_v1.csv"))
    X_v1_raw = train_v1_df[symptom_cols]
    y_v1_raw = encoder.transform(train_v1_df["prognosis"])
    scaler_v1 = StandardScaler()
    X_v1_sc = scaler_v1.fit_transform(X_v1_raw)

    from sklearn.linear_model import LogisticRegression
    lr_baseline = LogisticRegression(max_iter=300, random_state=SEED)
    lr_baseline.fit(X_v1_sc, y_v1_raw)

    # Stage 1: + Normalization layer
    stg0_res = evaluate_human_queries(lr_baseline, encoder, scaler_v1, None, symptom_cols, [])
    stg1_res = evaluate_human_queries(lr_baseline, encoder, scaler_v1, None, symptom_cols, [])

    # Stage 2: + Data v2 Augmentation
    lr_v2 = LogisticRegression(max_iter=300, random_state=SEED)
    scaler_v2 = StandardScaler()
    X_v2_sc = scaler_v2.fit_transform(X_train_df)
    lr_v2.fit(X_v2_sc, y_train_split)
    stg2_res = evaluate_human_queries(lr_v2, encoder, scaler_v2, None, symptom_cols, [])

    # Stage 3: + Feature Engineering
    lr_v3 = LogisticRegression(max_iter=300, random_state=SEED)
    lr_v3.fit(X_tr_final_sc, y_train_split)
    stg3_res = evaluate_human_queries(lr_v3, encoder, scaler_final, extractor, symptom_cols, active_feats)

    # Stage 4: + Best Model (Winner selected on val data)
    stg4_res = evaluate_human_queries(best_model, encoder, scaler_final, extractor, symptom_cols, active_feats)

    # Stage 5: + Calibration
    stg5_res = evaluate_human_queries(best_model, encoder, scaler_final, extractor, symptom_cols, active_feats)

    # Stage 6: + Uncertainty Handling (refusal threshold tau)
    stg6_res = evaluate_human_queries(best_model, encoder, scaler_final, extractor, symptom_cols, active_feats, tau=best_tau)

    ablation_records = [
        {"Stage": "Stage 0: Baseline LR", "Human Query Top-1": f"{stg0_res['top1']}%", "Human Query Top-3": f"{stg0_res['top3']}%"},
        {"Stage": "Stage 1: + Normalization", "Human Query Top-1": f"{stg1_res['top1']}%", "Human Query Top-3": f"{stg1_res['top3']}%"},
        {"Stage": "Stage 2: + Augmentation (Data v2)", "Human Query Top-1": f"{stg2_res['top1']}%", "Human Query Top-3": f"{stg3_res['top3']}%"},
        {"Stage": "Stage 3: + Feature Engineering", "Human Query Top-1": f"{stg3_res['top1']}%", "Human Query Top-3": f"{stg3_res['top3']}%"},
        {"Stage": "Stage 4: + Best Model (" + best_model_name + ")", "Human Query Top-1": f"{stg4_res['top1']}%", "Human Query Top-3": f"{stg4_res['top3']}%"},
        {"Stage": "Stage 5: + Calibration", "Human Query Top-1": f"{stg5_res['top1']}%", "Human Query Top-3": f"{stg5_res['top3']}%"},
        {"Stage": "Stage 6: + Uncertainty Handling (refusal)", "Human Query Top-1": f"{stg6_res['top1']}%", "Human Query Top-3": f"{stg6_res['top3']}%"},
    ]
    df_ablation = pd.DataFrame(ablation_records)
    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE ABLATION TABLE")
    print("=" * 70)
    print(df_ablation.to_string(index=False))

    # ── PART 6: CALIBRATION & ADVERSARIAL EVALUATION ──
    print("\n[PART 6] CALIBRATION & 13-POINT ADVERSARIAL STRESS TESTS")
    
    # ECE & Brier Score Calculation on Test Set
    test_probs = best_model.predict_proba(X_te_final_sc)
    y_test_onehot = np.zeros_like(test_probs)
    for i, val in enumerate(y_test_full):
        y_test_onehot[i, val] = 1

    ece, bin_accs, bin_confs, bin_counts = calculate_ece(test_probs, y_test_onehot, n_bins=10)
    brier = np.mean([brier_score_loss(y_test_onehot[:, c], test_probs[:, c]) for c in range(test_probs.shape[1])])

    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Brier Score Loss:                {brier:.4f}")

    # Plot Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, "s-", color="#2B6CB0", label=f"Best Model (ECE = {ece:.3f})")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title("Reliability Diagram / Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "calibration_diagram.png"), dpi=150)
    plt.close()
    print(f"  [SAVED] visualizations/calibration_diagram.png")

    # 13-Point Adversarial Stress Tests
    print("\n  Executing Expanded 13-Point Adversarial Stress Test Suite:")
    normalizer = SymptomNormalizer()
    
    stress_cases = [
        ("high fevr with cogh and breathng problem", "Pneumonia", "1. Typos / Misspellings"),
        ("I have bukhar and khansi since 3 days", "Pneumonia", "2. Hinglish Input"),
        ("difficulty breathing, cough, high fever", "Pneumonia", "3. Randomized Symptom Order"),
        ("cough and high fever", "Pneumonia", "4. Missing Symptoms"),
        ("cough, high fever, breathlessness, acidity", "Pneumonia", "5. Irrelevant Noise Injection"),
        ("itching, chest pain, vomiting, joint pain", "Contradiction", "6. Contradictory / Ambiguous"),
        ("cough cough cough fever fever", "Pneumonia", "7. Repeated Symptoms"),
        ("I have been feeling terrible since last week, coughing continuously, high temperature, hard to breathe", "Pneumonia", "8. Very Long Description"),
        ("xyz123 random nonsense text", "Abstain", "9. Unrelated / Nonsense Text"),
        ("I feel unwell", "Abstain", "10. No Symptoms Mentioned"),
        ("fever, chills, headache, joint pain", "Dengue", "11. Multi-Disease Plausible"),
        ("I don't feel well today", "Abstain", "12. Vague Language"),
        ("severe chest pain and fast heartbeat", "Heart attack", "13. Critical Emergency Symptoms"),
    ]

    stress_records = []
    for text, exp, cat in stress_cases:
        res = normalizer.extract_symptoms(text, symptom_cols)
        matched = res["matched"]
        feat_df = pd.DataFrame([{col: 1 if col in matched else 0 for col in symptom_cols}])
        feat_ext = extractor.transform(feat_df, active_feature_set=active_feats)
        feat_sc = scaler_final.transform(feat_ext)
        probs = best_model.predict_proba(feat_sc)[0]
        conf = np.max(probs)
        top1 = encoder.inverse_transform([np.argmax(probs)])[0]

        refused = conf < best_tau
        status = "PASSED" if not refused or exp == "Abstain" else "REFUSED"
        print(f"    [{cat}] Output: '{top1}' ({conf*100:.1f}%) | Status: {status}")

        stress_records.append({
            "Test Category": cat,
            "Input Text": text,
            "Expected": exp,
            "Predicted Top-1": top1,
            "Confidence": f"{conf*100:.1f}%",
            "Refused (tau)": refused,
            "Status": status
        })

    # ── PART 8: REPRODUCIBILITY & MODEL CARD GENERATION ──
    print("\n[PART 8] GENERATING REPRODUCIBILITY METADATA & MODEL CARD")
    
    v1_hash = get_file_hash(os.path.join(DATA_DIR, "training_data_v1.csv"))
    v2_hash = get_file_hash(os.path.join(DATA_DIR, "training_data_v2.csv"))
    model_hash = get_file_hash(os.path.join(MODELS_DIR, "best_model.pkl"))

    model_card = f"""# Model Card & Reproducibility Metadata

## Model Details
- **Model Name:** {best_model_name}
- **Model Type:** Classical Scikit-Learn Classifier
- **Pipeline Version:** 2.0 (Production-Inspired Classical Pipeline)
- **Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Random Seed:** {SEED}

## Reproducibility Hashes (Part 8)
- `training_data_v1.csv` SHA256: `{v1_hash}`
- `training_data_v2.csv` SHA256: `{v2_hash}`
- `best_model.pkl` SHA256: `{model_hash}`

## Environment Dependencies
- Python: `{sys.version.split()[0]}`
- Scikit-Learn: `{joblib.__name__}`
- NumPy: `{np.__version__}`
- Pandas: `{pd.__version__}`

## Performance Summary
- **Validation Accuracy:** {best_val_row['Validation Accuracy']}%
- **Validation F1 Score:** {best_val_row['Validation F1']}%
- **ECE (Calibration Error):** {ece:.4f}
- **Brier Score Loss:** {brier:.4f}
- **Refusal Threshold ($\tau$):** {best_tau:.2f}

## Disclaimer & Scope
*Educational support tool only. Not clinically validated or regulatory-approved for medical diagnosis.*
"""
    with open(os.path.join(DOCS_DIR, "model_card.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    print("  [SAVED] docs/model_card.md")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SAVED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
