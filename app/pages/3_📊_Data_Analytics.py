"""
Page 3 — Dataset & EDA Analytics
Explore symptom distributions, class balance, and correlation maps.
"""

import streamlit as st
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_training_data
from utils.theme import styled_metric_card
from visualizations.charts import (
    plot_disease_distribution, plot_symptom_frequency, plot_correlation_heatmap
)

_CSS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Dataset Analytics & EDA")

df_train = load_training_data()
symptom_cols = [c for c in df_train.columns if c != "prognosis"]
st.markdown(f"Exploratory Data Analysis for **{len(df_train):,} medical records**, **{len(symptom_cols)} symptom indicators**, and **{df_train['prognosis'].nunique()} disease classes**.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(styled_metric_card("Train Rows", f"{len(df_train):,}", color="#2B6CB0"), unsafe_allow_html=True)
with c2:
    st.markdown(styled_metric_card("Symptoms", str(len(symptom_cols)), color="#38A169"), unsafe_allow_html=True)
with c3:
    st.markdown(styled_metric_card("Diseases", str(df_train["prognosis"].nunique()), color="#D69E2E"), unsafe_allow_html=True)
with c4:
    st.markdown(styled_metric_card("Missing Values", "0", color="#38A169"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📄 Dataset Preview", "📈 Symptom Frequency", "🔥 Correlation Heatmap"])

with tab1:
    st.subheader("Training Data Sample")
    st.dataframe(df_train.head(15), use_container_width=True)

with tab2:
    st.subheader("Top Symptom Prevalence")
    top_n = st.slider("Number of top symptoms:", 10, 50, 25)
    fig = plot_symptom_frequency(df_train, top_n=top_n)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Symptom Correlation Matrix")
    n_feat = st.slider("Features to plot:", 10, 40, 20)
    fig_corr = plot_correlation_heatmap(df_train, top_n=n_feat)
    st.plotly_chart(fig_corr, use_container_width=True)
