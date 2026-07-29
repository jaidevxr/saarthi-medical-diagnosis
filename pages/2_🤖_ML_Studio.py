"""
Page 2 — Machine Learning Studio
Benchmark and switch between 6 ML algorithms.
"""

import streamlit as st
import os, sys
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.theme import styled_metric_card
from utils.data_loader import load_training_data
from visualizations.charts import plot_model_comparison

_CSS = os.path.join(PROJECT_ROOT, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

df_train = load_training_data()
st.title("🤖 ML Model Studio & Benchmarks")
st.markdown(f"Benchmark performance across **6 classical ML classification algorithms** trained on **{len(df_train):,} medical records** across **{df_train['prognosis'].nunique()} disease classes**.")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
results_path = os.path.join(MODELS_DIR, "model_results.csv")

if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    if "Algorithm" not in results_df.columns and "Model" in results_df.columns:
        results_df["Algorithm"] = results_df["Model"]
    if "Model" not in results_df.columns and "Algorithm" in results_df.columns:
        results_df["Model"] = results_df["Algorithm"]
        
    algo_col = "Algorithm" if "Algorithm" in results_df.columns else ("Model" if "Model" in results_df.columns else results_df.columns[0])

    st.subheader("Algorithm Evaluation Summary")
    st.dataframe(
        results_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#D4E8D4"),
        use_container_width=True,
    )

    best_row = results_df.loc[results_df["Accuracy"].idxmax()]

    acc_val = float(best_row['Accuracy'])
    acc_str = f"{acc_val:.2%}" if acc_val <= 1.0 else f"{acc_val/100.0:.2%}"
    
    f1_val = float(best_row['F1 Score'])
    f1_str = f"{f1_val:.4f}" if f1_val <= 1.0 else f"{f1_val/100.0:.4f}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(styled_metric_card("Top Classifier", str(best_row[algo_col]), color="#38A169"), unsafe_allow_html=True)
    with c2:
        st.markdown(styled_metric_card("Accuracy", acc_str, color="#2B6CB0"), unsafe_allow_html=True)
    with c3:
        st.markdown(styled_metric_card("F1 Score", f1_str, color="#D69E2E"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Performance Visual Comparison")

    fig = plot_model_comparison(results_df, metric_cols=["Accuracy", "Precision", "Recall", "F1 Score"])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("⚙️ Select Active Prediction Model")
    raw_models = results_df[algo_col].tolist()
    all_models = ["Production Multi-Turn Engine"] + [m for m in raw_models if m != "Production Multi-Turn Engine"]
    best_file = os.path.join(MODELS_DIR, "best_model_name.pkl")
    current_best = joblib.load(best_file) if os.path.exists(best_file) else all_models[0]

    chosen_model = st.selectbox("Active prediction model:", all_models, index=all_models.index(current_best) if current_best in all_models else 0)

    if st.button("Set Active Model"):
        joblib.dump(chosen_model, best_file)
        
        model_filename_map = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "Random Forest": "random_forest.pkl",
            "KNN": "knn.pkl",
            "SVC": "svm.pkl",
            "SVM": "svm.pkl",
            "Naive Bayes": "naive_bayes.pkl",
            "Calibrated Naive Bayes": "naive_bayes.pkl",
        }
        if chosen_model in model_filename_map:
            src_model = os.path.join(MODELS_DIR, model_filename_map[chosen_model])
            if os.path.exists(src_model):
                import shutil
                shutil.copy2(src_model, os.path.join(MODELS_DIR, "best_model.pkl"))

        st.cache_resource.clear()
        st.success(f"Active prediction model updated to: **{chosen_model}**")
else:
    st.info("No benchmark results found. Run pretraining script or check models directory.")
