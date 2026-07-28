"""
Permanent Unit Tests: Model Robustness, Order Invariance & Overconfidence Protection
=====================================================================================
Validates that model predictions are stable against:
1. Symptom ordering variations.
2. Missing symptoms / partial inputs.
3. Irrelevant symptom noise injections (e.g. +acidity doesn't flip Pneumonia).
4. Adversarial overconfidence protection (spread of probabilities).
"""

import os, sys, joblib
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

@pytest.fixture
def model_artifacts():
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    symptom_cols = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))
    
    model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "best_model_name.pkl")) else "best_model"
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "naive_bayes.pkl")
        
    model = joblib.load(model_path)
    return encoder, scaler, symptom_cols, model

def predict_top_differential(symptoms_list, artifacts, top_k=3):
    encoder, scaler, symptom_cols, model = artifacts
    feat = pd.DataFrame([[0] * len(symptom_cols)], columns=symptom_cols)
    for s in symptoms_list:
        if s in feat.columns:
            feat[s] = 1
            
    scaled = scaler.transform(feat)
    probs = model.predict_proba(scaled)[0]
    
    top_indices = np.argsort(probs)[-top_k:][::-1]
    diseases = encoder.inverse_transform(top_indices)
    top_probs = probs[top_indices]
    
    return list(zip(diseases, top_probs))

def test_symptom_order_invariance(model_artifacts):
    """Test that symptom vector encoding is 100% order invariant."""
    set_a = ["cough", "high_fever", "breathlessness", "chest_pain"]
    set_b = ["chest_pain", "breathlessness", "high_fever", "cough"]
    
    res_a = predict_top_differential(set_a, model_artifacts)
    res_b = predict_top_differential(set_b, model_artifacts)
    
    assert res_a[0][0] == res_b[0][0], f"Order invariance failed: {res_a[0][0]} vs {res_b[0][0]}"
    assert abs(res_a[0][1] - res_b[0][1]) < 1e-4, "Confidence score differs on reordered input!"

def test_irrelevant_noise_stability(model_artifacts):
    """Test that adding 1 irrelevant symptom does not catastrophically flip Pneumonia to GERD."""
    base_syms = ["cough", "high_fever", "chest_pain", "breathlessness"]
    base_res = predict_top_differential(base_syms, model_artifacts, top_k=3)
    base_disease = base_res[0][0] # Should be Pneumonia or Respiratory
    
    noisy_syms_1 = base_syms + ["acidity"]
    noisy_res_1 = predict_top_differential(noisy_syms_1, model_artifacts, top_k=3)
    noisy_diseases_1 = [d for d, _ in noisy_res_1]
    
    # Pneumonia must stay in top 3 differential diagnoses even when +acidity is injected!
    assert base_disease in noisy_diseases_1, (
        f"Irrelevant symptom noise ('acidity') dropped primary diagnosis {base_disease} out of top 3! "
        f"New top 3: {noisy_diseases_1}"
    )

def test_missing_symptom_resilience(model_artifacts):
    """Test that dropping 1 symptom from a classic profile preserves the top differential diagnosis."""
    full_pneumonia = ["cough", "high_fever", "breathlessness", "chest_pain", "phlegm"]
    full_res = predict_top_differential(full_pneumonia, model_artifacts, top_k=3)
    full_disease = full_res[0][0]
    
    # Drop 'chest_pain'
    partial_syms = ["cough", "high_fever", "breathlessness", "phlegm"]
    partial_res = predict_top_differential(partial_syms, model_artifacts, top_k=3)
    partial_diseases = [d for d, _ in partial_res]
    
    assert full_disease in partial_diseases, (
        f"Dropping 1 symptom from full profile lost primary diagnosis {full_disease}! "
        f"New top 3: {partial_diseases}"
    )

def test_overconfidence_protection(model_artifacts):
    """Test that contradictory or multi-disease inputs return a spread of probabilities rather than 100% certainty."""
    contradictory_syms = ["itching", "chest_pain", "vomiting", "joint_pain", "breathlessness"]
    results = predict_top_differential(contradictory_syms, model_artifacts, top_k=3)
    
    top_confidence = results[0][1] * 100
    # Overconfidence check: top confidence on contradictory input should be calibrated (< 99.9%)
    assert top_confidence < 99.9, f"Model exhibited 100% overconfidence ({top_confidence:.2f}%) on contradictory input!"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
