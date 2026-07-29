import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Medical Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

# Load saved model & artifacts
def load_saved_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(base_dir, 'saved_models')
    
    model_path = os.path.join(saved_dir, 'best_model.joblib')
    encoder_path = os.path.join(saved_dir, 'label_encoder.joblib')
    features_path = os.path.join(saved_dir, 'feature_names.joblib')
    symptoms_path = os.path.join(saved_dir, 'symptoms.json')
    comparison_path = os.path.join(saved_dir, 'model_comparison.json')

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    label_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    feature_names = joblib.load(features_path) if os.path.exists(features_path) else []
    
    symptoms_meta = []
    if os.path.exists(symptoms_path):
        with open(symptoms_path, 'r') as f:
            symptoms_meta = json.load(f)
            
    comparison_df = None
    if os.path.exists(comparison_path):
        comparison_df = pd.read_json(comparison_path)
        
    return model, label_encoder, feature_names, symptoms_meta, comparison_df

model, label_encoder, feature_names, symptoms_meta, comparison_df = load_saved_artifacts()

# App Header
st.title("🩺 Medical Disease Predictor")
st.markdown("Select patient symptoms below to predict the target medical condition using the trained machine learning model.")

# Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Disease Predictor", "📊 Model Comparison & Leaderboard"])

# TAB 1: DISEASE PREDICTOR
with tab1:
    st.subheader("Patient Symptom Selection")
    
    if model is None or label_encoder is None or len(feature_names) == 0:
        st.error("⚠️ Model artifacts missing. Please run `data_cleaning_and_modeling.ipynb` first.")
    else:
        symptom_map = {item['display']: item['raw'] for item in symptoms_meta}
        display_options = sorted(list(symptom_map.keys()))
        
        selected_displays = st.multiselect(
            "Select all present symptoms:",
            options=display_options,
            placeholder="Search symptoms (e.g. Skin Rash, High Fever, Vomiting, Joint Pain)..."
        )
        
        if st.button("Predict Disease"):
            if not selected_displays:
                st.error("Please select at least 1 symptom.")
            else:
                raw_selected = [symptom_map[d] for d in selected_displays]
                input_vector = np.zeros((1, len(feature_names)))
                
                for s in raw_selected:
                    if s in feature_names:
                        idx = feature_names.index(s)
                        input_vector[0, idx] = 1.0
                
                # Predict
                pred_encoded = model.predict(input_vector)[0]
                pred_disease = label_encoder.inverse_transform([pred_encoded])[0]
                
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color: #38bdf8; margin: 0;">Predicted Diagnosis: {pred_disease}</h2>
                    <p style="color: #94a3b8; margin-top: 6px;">Model Used: <strong>{type(model).__name__}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                if hasattr(model, "predict_proba"):
                    st.markdown("### Top Probability Confidence Breakdown")
                    probs = model.predict_proba(input_vector)[0]
                    top_3_idx = np.argsort(probs)[::-1][:3]
                    top_diseases = label_encoder.inverse_transform(top_3_idx)
                    top_probs = probs[top_3_idx]
                    
                    for dis, pr in zip(top_diseases, top_probs):
                        st.write(f"**{dis}**: `{pr * 100:.1f}%`")
                        st.progress(float(pr))

# TAB 2: MODEL COMPARISON
with tab2:
    st.subheader("Machine Learning Models Evaluation & Overfitting Check")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['Train Accuracy'], width, label='Train Accuracy', color='#38bdf8')
        ax.bar(x + width/2, comparison_df['Test Accuracy'], width, label='Test Accuracy', color='#34d399')
        
        ax.set_ylabel('Accuracy Score', color='white')
        ax.set_title('Train vs Test Accuracy (Overfitting Check)', color='white', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=15, color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
        ax.set_ylim(0.8, 1.02)
        
        st.pyplot(fig)
