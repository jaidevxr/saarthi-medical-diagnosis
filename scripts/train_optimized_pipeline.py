"""
Saarthi Medical Diagnosis — Model Training & Evaluation Script
================================================================
Trains and compares 6 classification models from the ML curriculum:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
  6. Naive Bayes (GaussianNB)

Evaluates each using: Accuracy, Precision, Recall, F1 Score, ROC-AUC
Selects the best model and saves it for production use.

Uses: Python, NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Scikit-Learn imports (Module 4: ML Fundamentals)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Classification Models (Module 4)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Visualization (Module 2)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")

# Create directories if they don't exist
for d in [MODELS_DIR, REPORTS_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING & PREPROCESSING (Module 1 & 2)
# ═══════════════════════════════════════════════════════════════════

def load_and_preprocess_data():
    """
    Load the medical symptom dataset and perform preprocessing.
    - Loads Training.csv and Testing.csv
    - Separates features (symptoms) from target (disease)
    - Encodes labels using LabelEncoder
    - Scales features using StandardScaler
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 70)

    # Load datasets using Pandas (Module 1)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "Testing.csv"))

    print(f"  Training dataset shape: {train_df.shape}")
    print(f"  Testing dataset shape:  {test_df.shape}")

    # Separate features and target
    symptom_cols = [col for col in train_df.columns if col != "prognosis"]
    X_train = train_df[symptom_cols]
    y_train = train_df["prognosis"]
    X_test = test_df[symptom_cols]
    y_test = test_df["prognosis"]

    print(f"  Number of symptoms (features): {len(symptom_cols)}")
    print(f"  Number of diseases (classes):  {y_train.nunique()}")

    # Data Quality Checks (Module 2: Data Preprocessing)
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    duplicates = train_df.duplicated().sum()
    print(f"  Missing values (train): {missing_train}")
    print(f"  Missing values (test):  {missing_test}")
    print(f"  Duplicate rows (train): {duplicates}")

    # Encode target labels (Module 1: Variables & Data Types)
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Feature Scaling using StandardScaler (Module 2: Data Transformation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Label encoding complete: {len(encoder.classes_)} classes")
    print(f"  Feature scaling complete (StandardScaler)")

    return (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded,
            encoder, scaler, symptom_cols, X_train, y_train)


# ═══════════════════════════════════════════════════════════════════
# STEP 2: DATA AUGMENTATION (Module 1: Loops & Functions)
# ═══════════════════════════════════════════════════════════════════

def augment_training_data(X_train_df, y_train_series, symptom_cols):
    """
    Augment training data by creating partial symptom variants.
    
    Why: Users describe 2-3 symptoms, but training data has 7-15 per disease.
    We create new training samples with fewer symptoms so the model
    learns to diagnose from partial inputs.
    
    Uses: NumPy arrays, Pandas DataFrames, loops, random sampling
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA AUGMENTATION")
    print("=" * 70)

    np.random.seed(42)

    X_mat = X_train_df.values
    y_arr = y_train_series.values

    augmented_X = []
    augmented_y = []

    for i in range(len(X_mat)):
        original_row = X_mat[i].copy()
        disease = y_arr[i]

        # Keep original sample
        augmented_X.append(original_row)
        augmented_y.append(disease)

        # Find active symptom indices (where value == 1)
        active_indices = np.where(original_row == 1)[0]
        n_active = len(active_indices)

        if n_active <= 2:
            continue

        # Create partial-symptom variants using loops (Module 1)
        # Variant 1: Keep only 2 random symptoms (5 variants)
        for _ in range(5):
            variant = np.zeros_like(original_row)
            keep_idx = np.random.choice(active_indices, size=min(2, n_active), replace=False)
            variant[keep_idx] = 1
            augmented_X.append(variant)
            augmented_y.append(disease)

        # Variant 2: Keep only 3 random symptoms (2 variants)
        if n_active > 3:
            for _ in range(2):
                variant = np.zeros_like(original_row)
                keep_idx = np.random.choice(active_indices, size=3, replace=False)
                variant[keep_idx] = 1
                augmented_X.append(variant)
                augmented_y.append(disease)

        # Variant 3: Keep 4-5 symptoms (1 variant)
        if n_active > 5:
            variant = np.zeros_like(original_row)
            keep_idx = np.random.choice(active_indices, size=min(4, n_active), replace=False)
            variant[keep_idx] = 1
            augmented_X.append(variant)
            augmented_y.append(disease)

        # Variant 4: Drop 1 random symptom (1 variant)
        variant = original_row.copy()
        drop_count = min(1, n_active - 1)
        drop_idx = np.random.choice(active_indices, size=drop_count, replace=False)
        variant[drop_idx] = 0
        augmented_X.append(variant)
        augmented_y.append(disease)

    # Convert back to DataFrame (Module 1: Pandas)
    X_aug = pd.DataFrame(augmented_X, columns=symptom_cols)
    y_aug = pd.Series(augmented_y, name="prognosis")

    # Shuffle the augmented data
    shuffle_idx = np.random.permutation(len(X_aug))
    X_aug = X_aug.iloc[shuffle_idx].reset_index(drop=True)
    y_aug = y_aug.iloc[shuffle_idx].reset_index(drop=True)

    print(f"  Original training samples:  {len(X_mat)}")
    print(f"  Augmented training samples: {len(X_aug)}")
    print(f"  Expansion factor:           {len(X_aug) / len(X_mat):.1f}x")

    return X_aug, y_aug


# ═══════════════════════════════════════════════════════════════════
# STEP 3: MODEL TRAINING & COMPARISON (Module 4: ML Fundamentals)
# ═══════════════════════════════════════════════════════════════════

def train_and_compare_models(X_train, X_test, y_train, y_test, encoder):
    """
    Train 6 classification models and compare using standard metrics.
    
    Models from Module 4 syllabus:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. KNN
    5. Support Vector Machine (SVM)
    6. Naive Bayes (GaussianNB)
    
    Evaluation Metrics from Module 4:
    - Accuracy, Precision, Recall, F1 Score, ROC-AUC
    """
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING & COMPARING 6 CLASSIFICATION MODELS")
    print("=" * 70)

    # Define all 6 models from the syllabus
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42, n_jobs=1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=15, random_state=42, n_jobs=1
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "Naive Bayes": GaussianNB(),
    }

    # Store results for comparison
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()

        # For heavy models like SVM, sample if dataset is huge to prevent long fit times
        if name == "SVM" and len(X_train) > 10000:
            sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
            X_fit = X_train[sample_indices]
            y_fit = y_train[sample_indices]
        else:
            X_fit = X_train
            y_fit = y_train

        # Train the model (Module 4: Training)
        model.fit(X_fit, y_fit)
        train_time = time.time() - start_time

        # Predict on test set (Module 4: Testing)
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics (Module 4: Model Evaluation)
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

        # ROC-AUC (requires probability predictions)
        try:
            y_prob = model.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted") * 100
        except Exception:
            roc = 0.0

        trained_models[name] = model

        print(f"    Accuracy:  {acc:.2f}%")
        print(f"    Precision: {prec:.2f}%")
        print(f"    Recall:    {rec:.2f}%")
        print(f"    F1 Score:  {f1:.2f}%")
        print(f"    ROC-AUC:   {roc:.2f}%")
        print(f"    Time:      {train_time:.1f}s")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 2),
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1 Score": round(f1, 2),
            "ROC-AUC": round(roc, 2),
            "Training Time (s)": round(train_time, 1),
        })

    # Create comparison DataFrame (Module 1: Pandas)
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return results_df, trained_models


# ═══════════════════════════════════════════════════════════════════
# STEP 4: HUMAN QUERY EVALUATION (Module 1: Functions, Loops)
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_human_queries(model, encoder, scaler, symptom_cols):
    """
    Evaluate the best model on real-world human symptom descriptions.
    Uses the NLP parser to convert natural text to symptom vectors.
    """
    print("\n" + "=" * 70)
    print("STEP 4: REAL-WORLD HUMAN QUERY EVALUATION")
    print("=" * 70)

    from utils.helpers import parse_symptoms_with_metadata

    # Load human benchmark queries
    queries_path = os.path.join(DATA_DIR, "human_eval_queries.json")
    if not os.path.exists(queries_path):
        print("  [SKIP] human_eval_queries.json not found")
        return None

    with open(queries_path, "r") as f:
        human_queries = json.load(f)

    top1_correct = 0
    top3_correct = 0

    for item in human_queries:
        text = item["text"]
        expected = item["expected"]

        # Parse symptoms from natural language (Module 1: String Processing)
        meta = parse_symptoms_with_metadata(text, symptom_cols)
        matched_symptoms = meta["matched"]

        # Build feature vector (Module 1: NumPy)
        feature_row = np.zeros(len(symptom_cols))
        for sym in matched_symptoms:
            if sym in symptom_cols:
                idx = symptom_cols.index(sym)
                feature_row[idx] = 1

        # Scale and predict
        feature_scaled = scaler.transform([feature_row])
        probs = model.predict_proba(feature_scaled)[0]

        # Get top-3 predictions
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_diseases = encoder.inverse_transform(top3_indices)

        # Check if correct
        if top3_diseases[0].lower().strip() == expected.lower().strip():
            top1_correct += 1
        if any(d.lower().strip() == expected.lower().strip() for d in top3_diseases):
            top3_correct += 1

    total = len(human_queries)
    top1_acc = top1_correct / total * 100
    top3_acc = top3_correct / total * 100

    print(f"  Total queries:        {total}")
    print(f"  Top-1 Accuracy:       {top1_acc:.2f}% ({top1_correct}/{total})")
    print(f"  Top-3 Accuracy:       {top3_acc:.2f}% ({top3_correct}/{total})")

    return {"top1_accuracy": round(top1_acc, 2), "top3_accuracy": round(top3_acc, 2)}


# ═══════════════════════════════════════════════════════════════════
# STEP 5: VISUALIZATIONS (Module 2: Matplotlib, Seaborn)
# ═══════════════════════════════════════════════════════════════════

def create_visualizations(results_df, best_model, X_test, y_test, encoder):
    """
    Create professional visualizations comparing model performance.
    Charts: Bar Chart, Heatmap (Confusion Matrix)
    """
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # ── Chart 1: Model Accuracy Comparison (Bar Chart) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    bars = ax.barh(results_df["Model"], results_df["Accuracy"], color=colors)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)

    # Add value labels on bars
    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "model_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  [SAVED] model_accuracy_comparison.png")

    # ── Chart 2: Multi-Metric Comparison (Grouped Bar Chart) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(results_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance: All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "multi_metric_comparison.png"), dpi=150)
    plt.close()
    print("  [SAVED] multi_metric_comparison.png")

    # ── Chart 3: Confusion Matrix Heatmap (Best Model) ──
    y_pred = best_model.predict(X_test)

    # Show top 20 diseases for readability
    unique_classes = encoder.classes_
    if len(unique_classes) > 20:
        # Get top 20 most frequent in test set
        top20 = pd.Series(encoder.inverse_transform(y_test)).value_counts().head(20).index
        mask = np.isin(encoder.inverse_transform(y_test), top20)
        y_test_sub = y_test[mask]
        y_pred_sub = y_pred[mask]
        labels = sorted(top20)
    else:
        y_test_sub = y_test
        y_pred_sub = y_pred
        labels = sorted(unique_classes)

    cm = confusion_matrix(
        encoder.inverse_transform(y_test_sub),
        encoder.inverse_transform(y_pred_sub),
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Disease", fontsize=12)
    ax.set_ylabel("Actual Disease", fontsize=12)
    ax.set_title("Confusion Matrix — Best Model (Top 20 Diseases)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confusion_matrix_heatmap.png"), dpi=150)
    plt.close()
    print("  [SAVED] confusion_matrix_heatmap.png")

    # ── Chart 4: ROC-AUC Comparison (Bar Chart) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(results_df["Model"], results_df["ROC-AUC"], color=colors)
    ax.set_xlabel("ROC-AUC Score (%)", fontsize=12)
    ax.set_title("ROC-AUC Score Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, results_df["ROC-AUC"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "roc_auc_comparison.png"), dpi=150)
    plt.close()
    print("  [SAVED] roc_auc_comparison.png")

    # ── Chart 5: Training Time Comparison (Bar Chart) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(results_df["Model"], results_df["Training Time (s)"], color=colors)
    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Model Training Time Comparison", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, results_df["Training Time (s)"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "training_time_comparison.png"), dpi=150)
    plt.close()
    print("  [SAVED] training_time_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# STEP 6: SAVE BEST MODEL & GENERATE REPORT (Module 1: File Handling)
# ═══════════════════════════════════════════════════════════════════

def save_best_model(results_df, trained_models, encoder, scaler, symptom_cols):
    """
    Select and save the best performing model based on accuracy.
    Uses File Handling (Module 1) and JSON export.
    """
    print("\n" + "=" * 70)
    print("STEP 6: SELECTING & SAVING BEST MODEL")
    print("=" * 70)

    # Select best model by accuracy (Module 4: Model Evaluation)
    best_row = results_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    print(f"  Best Model: {best_name}")
    print(f"  Accuracy:   {best_row['Accuracy']}%")
    print(f"  F1 Score:   {best_row['F1 Score']}%")

    # Save model artifacts using joblib (Module 1: File Handling)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(symptom_cols), os.path.join(MODELS_DIR, "symptom_columns.pkl"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    # Save model metadata as JSON (Module 1: File Handling)
    metadata = {
        "best_model_name": best_name,
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "f1_score": float(best_row["F1 Score"]),
        "roc_auc": float(best_row["ROC-AUC"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save results CSV
    results_df.to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)

    print(f"  [SAVED] best_model.pkl")
    print(f"  [SAVED] encoder.pkl, scaler.pkl, symptom_columns.pkl")
    print(f"  [SAVED] model_metadata.json")
    print(f"  [SAVED] model_results.csv")

    return best_name, best_model


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    """
    Main function: Orchestrates the full ML pipeline.
    Module 1: Functions, Module 4: ML Pipeline
    """
    print("\n")
    print("=" * 70)
    print("   SAARTHI MEDICAL DIAGNOSIS AI — MODEL TRAINING PIPELINE")
    print("   Using: Python, NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn")
    print("=" * 70)

    # Step 1: Load and preprocess data
    (X_train_scaled, X_test_scaled, y_train_enc, y_test_enc,
     encoder, scaler, symptom_cols, X_train_raw, y_train_raw) = load_and_preprocess_data()

    # Step 2: Augment training data with partial symptoms
    X_aug, y_aug = augment_training_data(X_train_raw, y_train_raw, symptom_cols)

    # Re-encode and re-scale augmented data
    y_aug_enc = encoder.transform(y_aug)
    scaler_aug = StandardScaler()
    X_aug_scaled = scaler_aug.fit_transform(X_aug)
    X_test_rescaled = scaler_aug.transform(
        pd.read_csv(os.path.join(DATA_DIR, "Testing.csv"))[symptom_cols]
    )

    # Step 3: Train and compare all 6 models
    results_df, trained_models = train_and_compare_models(
        X_aug_scaled, X_test_rescaled, y_aug_enc, y_test_enc, encoder
    )

    # Step 4: Save best model
    best_name, best_model = save_best_model(
        results_df, trained_models, encoder, scaler_aug, symptom_cols
    )

    # Step 5: Generate visualizations
    create_visualizations(results_df, best_model, X_test_rescaled, y_test_enc, encoder)

    # Step 6: Evaluate on human queries
    human_results = evaluate_on_human_queries(best_model, encoder, scaler_aug, symptom_cols)

    # Final Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Best Model:     {best_name}")
    print(f"  Test Accuracy:  {results_df.loc[results_df['Model']==best_name, 'Accuracy'].values[0]}%")
    if human_results:
        print(f"  Human Top-1:    {human_results['top1_accuracy']}%")
        print(f"  Human Top-3:    {human_results['top3_accuracy']}%")
    print(f"  Saved to:       models/best_model.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
