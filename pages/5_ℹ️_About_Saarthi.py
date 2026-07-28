"""
Saarthi — About & Project Documentation
=========================================
Overview of project architecture, medical taxonomy, contributors, and links.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

_FAVICON = os.path.join(PROJECT_ROOT, "assets", "favicon.png")
page_icon = _FAVICON if os.path.exists(_FAVICON) else "🩺"

st.set_page_config(
    page_title="About Saarthi — Clinical Decision Support",
    page_icon=page_icon,
    layout="wide",
)

_CSS_PATH = os.path.join(PROJECT_ROOT, "assets", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
        """
        <div style="background:#EBF8FF;border:1px solid #BEE3F8;border-radius:8px;padding:0.75rem;">
            <p style="margin:0;font-size:0.85rem;color:#2C5282 !important;font-weight:700;">
                📌 Project Info:
            </p>
            <p style="margin:0.3rem 0 0;font-size:0.82rem;color:#1A202C !important;">
                • 139 Disease Profiles<br>
                • 376 Symptom Features<br>
                • 12 Medical Domains
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Title Header ──
st.title("ℹ️ About Saarthi — Medical Diagnosis System")
st.markdown(
    "**Saarthi** is an advanced, production-grade Machine Learning clinical decision support platform "
    "designed to assist healthcare providers and individuals in generating instant differential diagnosis predictions."
)

st.divider()

# ── Mission & Key Highlights ──
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(
        """
        <div style="background:#FFFFFF;border:1px solid #CBD5E0;border-radius:12px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="color:#2B6CB0 !important;margin:0 0 0.5rem;">⚡ Instant Real-Time ML</h4>
            <p style="color:#2D3748 !important;font-size:0.9rem;margin:0;">
                Powered by a high-precision Random Forest Classifier (<b>99.6% Accuracy</b>) pre-compiled for instant inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        """
        <div style="background:#FFFFFF;border:1px solid #CBD5E0;border-radius:12px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="color:#2B6CB0 !important;margin:0 0 0.5rem;">💬 Advanced Natural Language NLP</h4>
            <p style="color:#2D3748 !important;font-size:0.9rem;margin:0;">
                Extracts clinical symptom indicators from natural conversational English input descriptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        """
        <div style="background:#FFFFFF;border:1px solid #CBD5E0;border-radius:12px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="color:#2B6CB0 !important;margin:0 0 0.5rem;">🏥 139 Disease Taxonomy</h4>
            <p style="color:#2D3748 !important;font-size:0.9rem;margin:0;">
                Covers 12 clinical specialty areas complete with specialist recommendations, precautions, and dietary guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Medical Specialties Covered ──
st.subheader("🩺 Medical Specialty Taxonomy (139 Diseases)")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        - **🫁 Respiratory Diseases**: Asthma, Pneumonia, COPD, Bronchitis, Tuberculosis, Sinusitis, Pleurisy, Croup
        - **🫀 Cardiovascular Disorders**: Heart Attack, Hypertension, Angina, DVT, Pericarditis, Peripheral Artery Disease
        - **🧠 Neurological Conditions**: Migraine, Epilepsy, Meningitis, Bell's Palsy, Sciatica, Trigeminal Neuralgia
        - **🧴 Skin & Dermatology**: Eczema, Psoriasis, Acne, Cellulitis, Shingles, Rosacea, Scabies, Ringworm
        - **🪵 Gastrointestinal Diseases**: GERD, Peptic Ulcer, Hepatitis A–E, IBS, Appendicitis, Pancreatitis, Gallstones
        - **🧫 Kidney & Urinary System**: UTI, Kidney Stones, Chronic Kidney Disease, Pyelonephritis, Prostatitis
        """
    )

with c2:
    st.markdown(
        """
        - **🩸 Endocrine & Metabolic**: Diabetes, Hypothyroidism, Hyperthyroidism, PCOS, Gout, Cushing's, Addison's
        - **🦠 Infectious Diseases**: Malaria, Dengue, Typhoid, Chickenpox, Measles, Lyme Disease, Tetanus, Cholera
        - **👂 ENT (Ear, Nose, Throat)**: Otitis Media, Tonsillitis, Meniere's, Tinnitus, Laryngitis, Nasal Polyps
        - **👁 Ophthalmology (Eye)**: Conjunctivitis, Glaucoma, Uveitis, Dry Eye Syndrome, Blepharitis, Corneal Abrasion
        - **🦴 Rheumatology & Autoimmune**: Osteoarthritis, Rheumatoid Arthritis, Lupus, Multiple Sclerosis, Fibromyalgia
        - **🩺 General & Emergency Medicine**: Anemia, Heat Stroke, Dehydration Syndrome, Chronic Fatigue Syndrome
        """
    )

st.divider()

# ── Official Project Links ──
st.subheader("🔗 Official Project Resources")

st.markdown(
    """
    - 🐙 **GitHub Repository**: [https://github.com/jaidevxr/saarthi-medical-diagnosis](https://github.com/jaidevxr/saarthi-medical-diagnosis)
    - 🌐 **Live Web Application**: [https://saarthi-medical-diagnosis.streamlit.app](https://saarthi-medical-diagnosis.streamlit.app)
    - 📄 **Documentation & Code**: Fully open-source under MIT License.
    """
)

st.divider()

# ── Medical Disclaimer ──
st.warning(
    "⚠️ **Medical Disclaimer:** Saarthi is designed as an educational decision support system powered by "
    "statistical machine learning algorithms. It is NOT a replacement for professional clinical diagnosis, advice, "
    "or treatment. Always consult a qualified medical professional for health concerns."
)
