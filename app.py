"""
Saarthi — Master Medical Diagnosis Prediction System (v7)
=====================================================
Streamlit Entry Point implementing Live Interactive Multi-Turn Dialogue Engine.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import MultiTurnDiagnosisEngine
from utils.data_loader import load_disease_info, symptom_display_name

# ── Favicon & Page Config ──
_FAVICON = os.path.join(os.path.dirname(__file__), "assets", "favicon.png")
page_icon = _FAVICON if os.path.exists(_FAVICON) else "🩺"

st.set_page_config(
    page_title="Saarthi — Multi-Turn Medical Diagnosis System",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ──
_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Styling Overrides ──
st.markdown("""
<style>
    .stTextArea textarea {
        border: 2px solid #CBD5E0 !important;
        border-radius: 10px !important;
        padding: 0.8rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #2B6CB0 !important;
        box-shadow: 0 0 0 2px rgba(43, 108, 176, 0.2) !important;
    }
    .question-card {
        background: #FEFCBF !important;
        border: 2px solid #D69E2E !important;
        border-radius: 14px !important;
        padding: 1.5rem 2rem !important;
        margin: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return MultiTurnDiagnosisEngine()

engine = get_engine()

# ── Sidebar Branding ──
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0;">
            <h2 style="margin:0;color:#2B6CB0 !important;font-weight:800;font-size:1.8rem;">🩺 Saarthi</h2>
            <p style="margin:0.2rem 0 0;font-size:0.85rem;color:#4A5568 !important;font-weight:600;">
                Multi-Turn Medical Diagnosis Assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style="background:#FFF5F5;border:1px solid #FEB2B2;border-radius:8px;padding:0.75rem;">
            <p style="margin:0;font-size:0.8rem;color:#9B2C2C !important;">
                <strong>Medical Disclaimer:</strong> Educational decision support tool. Always consult a licensed medical professional for clinical diagnosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Title Header ──
st.title("🩺 Saarthi — Multi-Turn Medical Diagnosis Assistant")
st.markdown("Describe how you feel below. If initial symptoms are ambiguous, Saarthi will ask up to 2 targeted clarifying questions to refine your diagnosis.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Initialize Session State ──
if "dialogue_state" not in st.session_state:
    st.session_state["dialogue_state"] = None

# Input Area
user_text = st.text_area(
    "Describe your symptoms in natural language:",
    placeholder="e.g., I have high fever with cough and difficulty breathing for 2 days...",
    height=110,
    key="initial_symptoms_input"
)

col_btn1, col_btn2 = st.columns([2, 5])
with col_btn1:
    start_clicked = st.button("🚀 Analyze Symptoms & Start Diagnosis", type="primary", use_container_width=True)

if start_clicked and user_text.strip():
    initial_res = engine.predict_initial(user_text)
    st.session_state["dialogue_state"] = initial_res

current_state = st.session_state.get("dialogue_state", None)

if current_state:
    st.divider()

    # Display Matched Symptoms
    matched = current_state.get("matched_symptoms", [])
    if matched:
        st.markdown("##### 🔬 Recognized Symptoms:")
        chips = " ".join([
            f'<span style="background:#2B6CB0;color:#FFFFFF !important;padding:0.35rem 0.8rem;'
            f'border-radius:16px;font-size:0.88rem;margin:0.15rem;display:inline-block;font-weight:600;">'
            f'{symptom_display_name(s)}</span>'
            for s in matched
        ])
        st.markdown(chips, unsafe_allow_html=True)

    # Check if a clarifying question is pending
    is_resolved = current_state.get("is_resolved", True)
    next_q_text = current_state.get("next_question_text", None)
    next_q_sym = current_state.get("next_question_symptom", None)
    rounds_asked = current_state.get("rounds_asked", 0)

    if not is_resolved and next_q_text and rounds_asked < 2:
        st.markdown(
            f"""
            <div class="question-card">
                <p style="margin:0;font-size:0.85rem;color:#744210 !important;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">
                    ❓ CLARIFYING QUESTION #{rounds_asked + 1} OF 2
                </p>
                <h3 style="margin:0.4rem 0 0.8rem;font-size:1.4rem;color:#1A202C !important;font-weight:700;">
                    {next_q_text}
                </h3>
                <p style="margin:0;font-size:0.85rem;color:#744210 !important;">
                    Answering this helps Saarthi distinguish between {len(current_state.get('candidate_diseases', []))} potential diseases.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_yes, col_no, _ = st.columns([1.5, 1.5, 5])
        with col_yes:
            yes_clicked = st.button("👍 YES, I have this symptom", key=f"q_yes_{rounds_asked}", use_container_width=True)
        with col_no:
            no_clicked = st.button("👎 NO, I do not have this", key=f"q_no_{rounds_asked}", use_container_width=True)

        if yes_clicked or no_clicked:
            ans = "YES" if yes_clicked else "NO"
            updated_res = engine.process_followup(
                present_symptoms=current_state["present_symptoms"],
                absent_symptoms=current_state["absent_symptoms"],
                asked_symptoms=current_state["asked_symptoms"],
                user_answer_yes_no=ans,
                current_question_symptom=next_q_sym,
                rounds_asked=rounds_asked
            )
            st.session_state["dialogue_state"] = updated_res
            st.rerun()

    else:
        # Final Diagnosis Output
        st.subheader("🎯 Diagnosis Results")
        top_d = current_state.get("primary_diagnosis", "Uncertain")
        candidates = current_state.get("candidate_diseases", [])
        top3 = current_state.get("top3_differential", [])

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
                <p style="margin:0;font-size:1.0rem;font-weight:600;color:#2B6CB0 !important;">
                    Resolved in {rounds_asked} clarifying follow-up question(s) | Candidates Remaining: {len(candidates)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Surface Tied Candidates if Ambiguity Remains
        if len(candidates) > 1:
            st.warning(f"⚠️ **Clinical Differential Notice:** Multiple diseases share high overlap with your symptoms: **{', '.join(candidates[:4])}**. Consult a specialist to differentiate.")

        # Render Top 3 Differential
        if top3:
            st.subheader("Top Differential Diagnoses")
            for rank, dis in enumerate(top3[:3]):
                st.markdown(f"**#{rank+1} {dis}**")

        # Render Clinical Info
        df_info = load_disease_info()
        info_matches = df_info[df_info["disease"] == top_d]
        if not info_matches.empty:
            info = info_matches.iloc[0]
            st.divider()
            st.subheader(f"🏥 Clinical Reference: {top_d}")
            st.markdown(f"**👨‍⚕️ Specialist:** `{info.get('specialist', 'General Physician')}`")
            st.markdown(f"**📋 Description:** {info.get('description', 'N/A')}")
            st.markdown(f"**🛡️ Prevention:** {info.get('prevention', 'N/A')}")
