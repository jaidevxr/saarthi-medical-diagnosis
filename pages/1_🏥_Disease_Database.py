"""
Page 1 — Master Disease Database
Explore and search diseases across 12+ categories with enriched metadata.
"""

import streamlit as st
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_disease_info, get_dataset_stats
from utils.theme import styled_metric_card

_CSS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🏥 Master Medical Knowledge Base")

df_info = load_disease_info()

try:
    stats = get_dataset_stats()
    symptom_count = stats["symptom_count"]
except Exception:
    symptom_count = 376

st.markdown(f"Comprehensive clinical reference data for **{len(df_info)} diseases** across **12+ medical specialties**.")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(styled_metric_card("Diseases", str(len(df_info)), color="#2B6CB0"), unsafe_allow_html=True)
with m2:
    num_cats = len(df_info["category"].unique()) if "category" in df_info.columns else 12
    st.markdown(styled_metric_card("Specialties", str(num_cats), color="#38A169"), unsafe_allow_html=True)
with m3:
    num_spec = len(df_info["specialist"].unique()) if "specialist" in df_info.columns else 10
    st.markdown(styled_metric_card("Specialists", str(num_spec), color="#D69E2E"), unsafe_allow_html=True)
with m4:
    st.markdown(styled_metric_card("Input Markers", f"{symptom_count} Symptoms", color="#805AD5"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Search & Filter
c_search, c_cat, c_sev = st.columns([2, 1.5, 1])
with c_search:
    search_q = st.text_input("🔍 Search Disease or Keyword", placeholder="e.g. Asthma, Malaria, Heart attack, Migraine...")
with c_cat:
    cats = ["All Categories"] + sorted(df_info["category"].unique().tolist()) if "category" in df_info.columns else ["All Categories"]
    selected_cat = st.selectbox("Category Filter", cats)
with c_sev:
    sevs = ["All Severities"] + sorted(df_info["severity"].unique().tolist()) if "severity" in df_info.columns else ["All Severities"]
    selected_sev = st.selectbox("Severity Filter", sevs)

filtered_df = df_info.copy()

if search_q:
    sq = search_q.lower()
    filtered_df = filtered_df[
        filtered_df["disease"].str.lower().str.contains(sq, na=False) |
        filtered_df["description"].str.lower().str.contains(sq, na=False) |
        filtered_df["specialist"].str.lower().str.contains(sq, na=False)
    ]

if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df["category"] == selected_cat]

if selected_sev != "All Severities":
    filtered_df = filtered_df[filtered_df["severity"] == selected_sev]

st.caption(f"Displaying **{len(filtered_df)}** of **{len(df_info)}** diseases.")

tab1, tab2 = st.tabs(["🗂️ Disease Inspector", "📊 Full Database Table"])

with tab1:
    if filtered_df.empty:
        st.warning("No diseases match your search criteria.")
    else:
        disease_names = sorted(filtered_df["disease"].tolist())
        selected_dis = st.selectbox("Select Disease:", disease_names)
        row = filtered_df[filtered_df["disease"] == selected_dis].iloc[0]

        sev = row.get('severity', 'Moderate')
        sev_color = "#E53E3E" if "High" in str(sev) or "Critical" in str(sev) else "#DD6B20" if "Moderate" in str(sev) else "#38A169"

        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:2px solid #2B6CB0;border-radius:10px;padding:1.4rem;margin:1rem 0;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <h2 style="margin:0;color:#1A365D !important;font-size:1.8rem;font-weight:800;">{row['disease']}</h2>
                <p style="margin:0.4rem 0 0;font-size:0.95rem;color:#4A5568 !important;font-weight:600;">
                    <strong>Category:</strong> {row.get('category', 'N/A')} &nbsp;|&nbsp; 
                    <strong>Specialist:</strong> {row.get('specialist', 'General Physician')} &nbsp;|&nbsp; 
                    <strong>Severity:</strong> <span style="color:{sev_color} !important;font-weight:700;">{sev}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick stats row
        qs1, qs2, qs3 = st.columns(3)
        with qs1:
            recovery = row.get('recovery_time', '')
            if pd.notna(recovery) and str(recovery) not in ["nan", ""]:
                st.metric("⏱️ Recovery Time", str(recovery))
        with qs2:
            contagious = row.get('contagious', '')
            if pd.notna(contagious) and str(contagious) not in ["nan", "", "Consult your doctor"]:
                st.metric("🦠 Contagious", str(contagious)[:30])
        with qs3:
            self_care = row.get('self_care', '')
            if pd.notna(self_care) and str(self_care) not in ["nan", "", "Consult your doctor", "Consult your doctor for personalized guidance."]:
                st.metric("🏠 Self-Care", str(self_care)[:40])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### 📋 Overview\n**Description:**\n{row.get('description', 'N/A')}\n\n**Common Causes:**\n{row.get('causes', 'N/A')}\n\n**Risk Factors:**\n{row.get('risk_factors', 'N/A')}")
        with col_b:
            st.markdown(f"### 🩺 Care Management\n**Specialist:**\n`{row.get('specialist', 'General Physician')}`\n\n**Diet:**\n{row.get('diet', 'N/A')}\n\n**Prevention:**\n{row.get('prevention', 'N/A')}")

        # Emergency warning signs
        emergency = row.get('emergency_signs', '')
        if pd.notna(emergency) and str(emergency) not in ["nan", ""]:
            st.markdown(
                f"""
                <div style="background:#FFF5F5;border:2px solid #FC8181;border-radius:10px;padding:1rem;margin-top:1rem;">
                    <p style="margin:0;font-size:0.9rem;color:#742A2A !important;font-weight:700;">
                        ⚠️ EMERGENCY WARNING SIGNS
                    </p>
                    <p style="margin:0.4rem 0 0;font-size:0.88rem;color:#9B2C2C !important;">
                        {emergency}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab2:
    st.dataframe(filtered_df, use_container_width=True, height=500)
