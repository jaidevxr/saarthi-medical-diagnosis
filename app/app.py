"""
Saarthi — Master Medical Diagnosis Prediction System (v2)
=====================================================
Main Streamlit entry point — Instant Diagnosis Engine.
Features:
- Natural Language Description as DEFAULT input method.
- Automatic real-time symptom detection and direct diagnosis prediction.
- Spelling correction & Hinglish support.
- Matched/unmatched symptom analysis.
- Symptom overlap explanation per disease.
- Enhanced disease info cards with recovery time, emergency signs, prevention.
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

from utils.data_loader import load_disease_info, symptom_display_name, load_disease_symptoms_map, get_dataset_stats
from utils.theme import styled_warning_box
from utils.helpers import parse_symptoms_from_text, parse_symptoms_with_metadata, save_prediction

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

@st.cache_resource
def load_artifacts():
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "encoder.pkl")) else joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")) else joblib.load(os.path.join(MODELS_DIR, "symptom_encoder.pkl"))
    symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "best_model_name.pkl")) else "Calibrated Naive Bayes"
    
    model_path = os.path.join(MODELS_DIR, "calibrated_nb.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    
    model = joblib.load(model_path)
    return encoder, scaler, symptom_columns, best_name, model

try:
    encoder, scaler, symptom_columns, active_model_name, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading ML model artifacts: {e}")
    st.stop()

# ── Dynamic Stats ──
try:
    stats = get_dataset_stats()
    disease_count = stats["disease_count"]
    symptom_count = stats["symptom_count"]
except Exception:
    disease_count = len(encoder.classes) if hasattr(encoder, 'classes') else 139
    symptom_count = len(symptom_columns)

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
                {disease_count} Diseases | {symptom_count} Symptoms
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:0.75rem;margin-bottom:1rem;">
            <p style="margin:0;font-size:0.82rem;color:#22543D !important;font-weight:700;">
                🌐 NLP Features:
            </p>
            <p style="margin:0.3rem 0 0;font-size:0.78rem;color:#2D3748 !important;">
                • Spelling auto-correction<br>
                • Hindi-English (Hinglish) input<br>
                • Layman's terms (sugar, BP, gas)<br>
                • Conversational natural language
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style="background:#F7FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:0.75rem;margin-bottom:1rem;">
            <p style="margin:0;font-size:0.82rem;color:#2D3748 !important;font-weight:700;">
                👥 Lead Contributors:
            </p>
            <p style="margin:0.3rem 0 0;font-size:0.82rem;">
                • <a href="https://github.com/KhushiSharma006" target="_blank" style="color:#3182CE !important;font-weight:600;text-decoration:none;">Khushi Sharma (@KhushiSharma006)</a><br>
                • <a href="https://github.com/jaidevxr" target="_blank" style="color:#3182CE !important;font-weight:600;text-decoration:none;">Jai (@jaidevxr)</a>
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
match_metadata = None  # Will store corrections/unmatched info

if "Natural Language" in input_mode:
    st.markdown("#### Describe Your Symptoms in Plain Text")
    st.caption("🌐 Supports: English • Hindi-English (bukhar, khansi, gale mein dard) • Layman terms (sugar, BP, gas) • Auto-corrects typos")
    user_text = st.text_area(
        "Enter your symptoms below:",
        placeholder="e.g., I have high fever, chills, severe headache, joint pain, and nausea for 2 days...\nor: bukhar, khansi, pet dard, chakkar aa raha hai...",
        height=130,
        key="nld_text_input",
    )
    if user_text:
        # Use enhanced parser with metadata
        match_metadata = parse_symptoms_with_metadata(user_text, symptom_columns)
        selected_symptoms = match_metadata["matched"]

        if selected_symptoms:
            run_directly = True

            # Show corrections applied (if any)
            if match_metadata["corrections"]:
                corrections_str = ", ".join([f"'{k}' → '{v}'" for k, v in match_metadata["corrections"].items()])
                st.markdown(
                    f'<div style="background:#FFFBEB;border:1px solid #F6E05E;border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.8rem;">'
                    f'<span style="font-size:0.85rem;color:#744210 !important;">✏️ <strong>Auto-corrected:</strong> {corrections_str}</span></div>',
                    unsafe_allow_html=True,
                )

            # Show Hinglish translations (if any)
            if match_metadata["hinglish_detected"]:
                hindi_str = ", ".join([f"'{h}'" for h in match_metadata["hinglish_detected"]])
                st.markdown(
                    f'<div style="background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.8rem;">'
                    f'<span style="font-size:0.85rem;color:#22543D !important;">🌐 <strong>Hinglish detected:</strong> {hindi_str}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**✅ Recognized Symptoms:**")
            chips = " ".join([
                f'<span style="background:#2B6CB0;color:#FFFFFF !important;padding:0.35rem 0.8rem;'
                f'border-radius:16px;font-size:0.88rem;margin:0.15rem;display:inline-block;font-weight:600;">'
                f'{symptom_display_name(s)}</span>'
                for s in selected_symptoms
            ])
            st.markdown(chips, unsafe_allow_html=True)

            # Show unmatched tokens (if any)
            if match_metadata["unmatched_tokens"]:
                unmatched_str = ", ".join([f"'{t}'" for t in match_metadata["unmatched_tokens"]])
                st.markdown(
                    f'<div style="background:#FFF5F5;border:1px solid #FEB2B2;border-radius:8px;padding:0.6rem 1rem;margin-top:0.8rem;">'
                    f'<span style="font-size:0.85rem;color:#742A2A !important;">❓ <strong>Unrecognized terms:</strong> {unmatched_str} '
                    f'<em>(these words didn\'t match any known symptoms)</em></span></div>',
                    unsafe_allow_html=True,
                )
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

    # ── Load disease symptom map for overlap analysis ──
    try:
        disease_symptom_map = load_disease_symptoms_map()
    except Exception:
        disease_symptom_map = {}

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

    # ── Top 5 Differential Diagnoses with Symptom Overlap ──
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

        # ── Symptom Overlap Explanation ──
        if dis in disease_symptom_map:
            known_symptoms = set(disease_symptom_map[dis])
            user_symptoms_set = set(selected_symptoms)
            overlap = user_symptoms_set & known_symptoms
            missing = known_symptoms - user_symptoms_set

            overlap_count = len(overlap)
            total_known = len(known_symptoms)
            overlap_pct = (overlap_count / total_known * 100) if total_known > 0 else 0

            # Build overlap explanation
            overlap_names = [symptom_display_name(s) for s in sorted(overlap)]
            missing_names = [symptom_display_name(s) for s in sorted(missing)][:5]  # Show top 5 missing

            with st.expander(f"📊 Why {dis}? — Matched {overlap_count}/{total_known} symptoms ({overlap_pct:.0f}%)"):
                if overlap_names:
                    overlap_chips = " ".join([
                        f'<span style="background:#C6F6D5;color:#22543D !important;padding:0.25rem 0.6rem;'
                        f'border-radius:12px;font-size:0.8rem;margin:0.1rem;display:inline-block;font-weight:600;">'
                        f'✓ {name}</span>'
                        for name in overlap_names
                    ])
                    st.markdown(f"**Matched symptoms:** {overlap_chips}", unsafe_allow_html=True)

                if missing_names:
                    missing_chips = " ".join([
                        f'<span style="background:#FED7D7;color:#742A2A !important;padding:0.25rem 0.6rem;'
                        f'border-radius:12px;font-size:0.8rem;margin:0.1rem;display:inline-block;font-weight:600;">'
                        f'✗ {name}</span>'
                        for name in missing_names
                    ])
                    remaining = len(known_symptoms - user_symptoms_set) - len(missing_names)
                    more_text = f" <em>+{remaining} more</em>" if remaining > 0 else ""
                    st.markdown(f"**You didn't mention:** {missing_chips}{more_text}", unsafe_allow_html=True)

    st.divider()

    # ── Disease Details ──
    df_info = load_disease_info()
    info_matches = df_info[df_info["disease"] == top_d]

    if not info_matches.empty:
        info = info_matches.iloc[0]
        st.subheader(f"🏥 Clinical Reference: {top_d}")

        # ── Quick Stats Row ──
        c_spec, c_sev, c_cat, c_rec = st.columns(4)
        with c_spec:
            st.markdown(f"**👨‍⚕️ Specialist:**\n`{info.get('specialist', 'General Physician')}`")
        with c_sev:
            sev = info.get('severity', 'Moderate')
            sev_color = "#E53E3E" if "High" in str(sev) or "Critical" in str(sev) else "#DD6B20" if "Moderate" in str(sev) else "#38A169"
            st.markdown(f"**⚡ Severity:**\n<span style='color:{sev_color} !important;font-weight:700;'>{sev}</span>", unsafe_allow_html=True)
        with c_cat:
            st.markdown(f"**🏷️ Category:**\n`{info.get('category', 'General')}`")
        with c_rec:
            recovery = info.get('recovery_time', 'Consult specialist')
            if pd.notna(recovery) and recovery != "nan":
                st.markdown(f"**⏱️ Recovery:**\n`{recovery}`")

        st.markdown(f"**📋 Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**🔬 Common Causes:** {info.get('causes', 'N/A')}")

        # ── Prevention ──
        prevention = info.get('prevention', 'N/A')
        if pd.notna(prevention) and prevention != "nan":
            st.markdown(f"**🛡️ Prevention:** {prevention}")

        # ── Contagious status ──
        contagious = info.get('contagious', '')
        if pd.notna(contagious) and contagious not in ["nan", "", "Consult your doctor"]:
            contagious_color = "#E53E3E" if "Yes" in str(contagious) else "#38A169"
            st.markdown(f"**🦠 Contagious:** <span style='color:{contagious_color} !important;font-weight:700;'>{contagious}</span>", unsafe_allow_html=True)

        # ── Self-care ──
        self_care = info.get('self_care', '')
        if pd.notna(self_care) and self_care not in ["nan", "", "Consult your doctor", "Consult your doctor for personalized guidance."]:
            st.markdown(f"**🏠 Self-care:** {self_care}")

        # ── Emergency Warning Signs ──
        emergency = info.get('emergency_signs', '')
        if pd.notna(emergency) and emergency != "nan":
            st.markdown(
                f"""
                <div style="background:#FFF5F5;border:2px solid #FC8181;border-radius:10px;padding:1rem;margin-top:1rem;">
                    <p style="margin:0;font-size:0.9rem;color:#742A2A !important;font-weight:700;">
                        ⚠️ EMERGENCY WARNING SIGNS — Seek immediate medical attention if:
                    </p>
                    <p style="margin:0.4rem 0 0;font-size:0.88rem;color:#9B2C2C !important;">
                        {emergency}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
