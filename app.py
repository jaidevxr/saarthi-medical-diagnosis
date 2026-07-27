"""
Saarthi — Master Medical Diagnosis Prediction System
=====================================================
Main Streamlit entry point — Instant Diagnosis Engine.
Features:
- Natural Language Description as DEFAULT input method.
- Automatic real-time symptom detection and direct diagnosis prediction.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os, sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Favicon Path ──
_FAVICON = os.path.join(os.path.dirname(__file__), "assets", "favicon.png")
page_icon = _FAVICON if os.path.exists(_FAVICON) else "🩺"

# ── Page Config ──
st.set_page_config(
    page_title="Saarthi — Medical Diagnosis System",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ──
_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.data_loader import load_disease_info, symptom_display_name
from utils.theme import styled_warning_box
from utils.helpers import parse_symptoms_from_text, save_prediction

# ── Load Pretrained Model Artifacts (Instant Load) ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

@st.cache_resource
def load_artifacts():
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "best_model_name.pkl")) else "Random Forest"
    
    model_filename = f"{best_name.lower().replace(' ', '_')}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    
    model = joblib.load(model_path)
    return encoder, scaler, symptom_columns, best_name, model

try:
    encoder, scaler, symptom_columns, active_model_name, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading ML model artifacts: {e}")
    st.stop()

# ── Sidebar Branding ──
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0;">
            <h2 style="margin:0;color:#2B6CB0 !important;font-weight:800;font-size:1.8rem;">🩺 Saarthi</h2>
            <p style="margin:0.2rem 0 0;font-size:0.85rem;color:#4A5568 !important;font-weight:600;">
                Medical Diagnosis Assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"""
        <div style="background:#EBF8FF;border:1px solid #BEE3F8;border-radius:8px;padding:0.75rem;margin-bottom:1rem;">
            <p style="margin:0;font-size:0.82rem;color:#2C5282 !important;font-weight:700;">
                🤖 Active Classifier:
            </p>
            <p style="margin:0.2rem 0 0;font-size:0.95rem;color:#1A202C !important;font-weight:700;">
                {active_model_name}
            </p>
            <p style="margin:0.2rem 0 0;font-size:0.78rem;color:#4A5568 !important;">
                139 Diseases | 376 Symptoms
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        styled_warning_box(
            "<strong>Medical Disclaimer:</strong> Educational decision support tool. "
            "Always consult a licensed medical professional for clinical diagnosis."
        ),
        unsafe_allow_html=True,
    )

# ── Header ──
st.title("🩺 Saarthi — Medical Diagnosis Prediction")
st.markdown("Describe how you feel or select symptoms below to generate an instant differential diagnosis prediction.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Mode Selector (DEFAULT = Natural Language Description) ──
input_mode = st.radio(
    "Choose Input Method:",
    ["💬 Natural Language Description", "📋 Structured Symptom Search"],
    index=0,
    horizontal=True,
)

selected_symptoms = []
run_directly = False

if "Natural Language" in input_mode:
    st.markdown("#### Describe Your Symptoms in Plain Text")
    user_text = st.text_area(
        "Enter your symptoms below:",
        placeholder="e.g., I have high fever, chills, severe headache, joint pain, and nausea for 2 days...",
        height=130,
        key="nld_text_input",
    )
    if user_text:
        selected_symptoms = parse_symptoms_from_text(user_text, symptom_columns)
        if selected_symptoms:
            run_directly = True
            st.markdown("**Recognized Symptoms:**")
            chips = " ".join([
                f'<span style="background:#2B6CB0;color:#FFFFFF !important;padding:0.35rem 0.8rem;'
                f'border-radius:16px;font-size:0.88rem;margin:0.15rem;display:inline-block;font-weight:600;">'
                f'{symptom_display_name(s)}</span>'
                for s in selected_symptoms
            ])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("💡 Type your symptoms in the box above (e.g. 'fever, headache, cough, itching, stomach pain').")
else:
    st.markdown("#### Select Symptoms from Database")
    display_names = sorted([symptom_display_name(s) for s in symptom_columns])
    chosen = st.multiselect(
        "Search and choose symptoms:",
        display_names,
        placeholder="Type symptom keywords (e.g. fever, headache, cough, chest pain, itching)...",
    )
    for name in chosen:
        for col in symptom_columns:
            if symptom_display_name(col).lower() == name.lower():
                selected_symptoms.append(col)
                break

st.markdown("<br>", unsafe_allow_html=True)

predict_disabled = len(selected_symptoms) == 0
btn_clicked = st.button("🚀 Analyze Symptoms & Predict Diagnosis", type="primary", disabled=predict_disabled, use_container_width=True)

# Run prediction directly if natural language description has detected symptoms OR if button clicked
if (run_directly or btn_clicked) and selected_symptoms:
    feature_vector = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)
    for sym in selected_symptoms:
        if sym in feature_vector.columns:
            feature_vector[sym] = 1

    feature_scaled = scaler.transform(feature_vector)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_scaled)[0]
    else:
        pred = model.predict(feature_scaled)[0]
        probs = np.zeros(len(encoder.classes))
        probs[pred] = 1.0

    top_5_idx = np.argsort(probs)[-5:][::-1]
    top_diseases = encoder.inverse_transform(top_5_idx)
    top_probs = probs[top_5_idx]

    matched_names = [symptom_display_name(s) for s in selected_symptoms]
    pred_records = [
        {"disease": d, "probability": round(float(p * 100), 1)}
        for d, p in zip(top_diseases, top_probs)
    ]
    save_prediction(matched_names, pred_records)

    top_d = top_diseases[0]
    top_p = top_probs[0] * 100

    # ── High Contrast White Result Card ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Prediction Results")

    st.markdown(
        f"""
        <div style="
            background: #FFFFFF !important;
            border: 2px solid #2B6CB0 !important;
            border-radius: 14px !important;
            padding: 1.6rem 2rem !important;
            box-shadow: 0 4px 14px rgba(43, 108, 176, 0.12) !important;
            margin-bottom: 1.5rem !important;
            text-align: center !important;
        ">
            <p style="margin:0;font-size:0.85rem;letter-spacing:0.08em;text-transform:uppercase;color:#718096 !important;font-weight:700;">
                PRIMARY DIFFERENTIAL DIAGNOSIS
            </p>
            <h2 style="margin:0.4rem 0;font-size:2.4rem;font-weight:800;color:#1A365D !important;letter-spacing:-0.02em;">
                {top_d}
            </h2>
            <p style="margin:0;font-size:1.25rem;font-weight:700;color:#276749 !important;">
                Confidence Score: {top_p:.1f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top 5 Differential Diagnoses ──
    st.subheader("Top 5 Differential Diagnoses")

    for i, (dis, pr) in enumerate(zip(top_diseases, top_probs)):
        p_pct = pr * 100
        col_name, col_bar, col_val = st.columns([3, 5, 1.2])

        with col_name:
            st.markdown(f"<span style='color:#1A202C !important;font-weight:700;'>#{i+1} {dis}</span>", unsafe_allow_html=True)
        with col_bar:
            st.progress(float(pr))
        with col_val:
            st.markdown(f"<span style='color:#2B6CB0 !important;font-weight:700;'>{p_pct:.1f}%</span>", unsafe_allow_html=True)

    st.divider()

    # ── Disease Details ──
    df_info = load_disease_info()
    info_matches = df_info[df_info["disease"] == top_d]

    if not info_matches.empty:
        info = info_matches.iloc[0]
        st.subheader(f"🏥 Clinical Reference Information: {top_d}")

        c_spec, c_sev, c_cat = st.columns(3)
        with c_spec:
            st.markdown(f"**Recommended Specialist:**\n`{info.get('specialist', 'General Physician')}`")
        with c_sev:
            st.markdown(f"**Severity Rating:**\n`{info.get('severity', 'Moderate')}`")
        with c_cat:
            st.markdown(f"**Specialty Category:**\n`{info.get('category', 'General')}`")

        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**Common Causes:** {info.get('causes', 'N/A')}")
        st.markdown(f"**Prevention & Precautions:** {info.get('prevention', 'N/A')}")
