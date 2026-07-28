"""
Page 2 — Machine Learning Studio
Benchmark and switch between 6 ML algorithms.
"""

import streamlit as st
import os, sys
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.theme import styled_metric_card
from visualizations.charts import plot_model_comparison

_CSS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🤖 ML Model Studio & Benchmarks")
st.markdown("Benchmark performance across **6 classical ML classification algorithms** trained on 20,850 records.")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
results_path = os.path.join(MODELS_DIR, "model_results.csv")

if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
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
    all_models = results_df[algo_col].tolist()
    best_file = os.path.join(MODELS_DIR, "best_model_name.pkl")
    current_best = joblib.load(best_file) if os.path.exists(best_file) else all_models[0]

    chosen_model = st.selectbox("Active prediction model:", all_models, index=all_models.index(current_best) if current_best in all_models else 0)

    if st.button("Set Active Model"):
        joblib.dump(chosen_model, best_file)
        st.success(f"Active prediction model updated to: **{chosen_model}**")
else:
    st.info("No benchmark results found. Run pretraining script or check models directory.")
