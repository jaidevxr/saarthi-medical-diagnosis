"""
Clinical Validation & Accuracy Test Suite (31 Real-World & Conversational Cases)
Tests the NLP parser + Random Forest classification pipeline against ground truth medical cases.
"""

import os, sys
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import parse_symptoms_from_text
from utils.data_loader import symptom_display_name

MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
symptom_columns = joblib.load(os.path.join(MODELS_DIR, "symptom_columns.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

CLINICAL_TEST_CASES = [
    ("High fever, shivering, severe chills, sweating, headache, nausea and muscle pain", "Malaria"),
    ("High fever, joint pain, rash, pain behind eyes, severe headache, back pain, red spots", "Dengue"),
    ("Cough with rust-colored phlegm, high fever, chills, chest pain, breathlessness, sweating", "Pneumonia"),
    ("Heartburn, acidity, chest pain, stomach pain, acid reflux, vomiting, regurgitation", "GERD"),
    ("One-sided severe headache, nausea, vomiting, sensitivity to light, sensitivity to sound", "Migraine"),
    ("Severe chest pain, chest pressure, sweating, breathlessness, nausea, anxiety, fainting", "Heart attack"),
    ("High fever, abdominal pain, headache, chills, constipation, diarrhoea, toxic look", "Typhoid"),
    ("Excessive thirst, frequent urination, fatigue, weight loss, excessive hunger, blurred vision", "Diabetes"),
    ("Cough, wheezing, breathlessness, chest tightness, coughing at night, fatigue", "Bronchial Asthma"),
    ("Persistent cough, blood in sputum, night sweats, high fever, weight loss, chronic fatigue", "Tuberculosis"),
    ("Itching, skin rash, blister, high fever, fatigue, lethargy, red spots over body", "Chicken pox"),
    ("Burning micturition, bladder discomfort, urinary urgency, foul smell of urine, cloudy urine", "Urinary tract infection"),
    ("Yellowish skin, yellowing of eyes, dark urine, itching, abdominal pain, vomiting, fatigue", "Jaundice"),
    ("Severe headache, dizziness, breathlessness, nosebleeds, blurred vision", "Hypertension"),
    ("Continuous sneezing, shivering, chills, watering from eyes, eye itching, congestion, hives", "Allergy"),
    ("Itching, skin rash, dry skin, skin cracking, skin flaking, skin thickening, oozing from skin", "Eczema"),
    ("Skin rash, skin peeling, silver-like dusting, small dents in nails, inflammatory nails, joint pain", "Psoriasis"),
    ("Severe joint pain, joint redness, joint warmth, swelling joints, foot pain, knee pain", "Gout"),
    ("Severe ear pain, ear discharge, hearing loss, high fever, ear fullness, balance problems", "Otitis Media"),
    ("Red eye, eye discharge, eye itching, watering from eyes, crusty eyelids, photophobia", "Conjunctivitis"),
    ("Severe right lower abdominal pain, nausea, vomiting, high fever, loss of appetite", "Appendicitis"),
    ("Fatigue, yellowish skin, dark urine, loss of appetite, abdominal pain, yellow eyes, lethargy, receiving unsterile injections", "Hepatitis B"),
    ("Fatigue, pale skin, breathlessness, dizziness, cold hands and feet, headache, weakness", "Anemia"),
    ("Vomiting, nausea, diarrhoea, abdominal cramping, high fever, dehydration, body aches", "Food Poisoning"),
    ("Headache, facial pain, facial pressure, congestion, nasal discharge, sinus pressure", "Sinusitis"),
    ("Sore throat, painful swallowing, high fever, swelled lymph nodes, white patches on throat", "Tonsillitis"),
    ("Skin rash, blister, skin burning, itching, high fever, fatigue, headache, skin tenderness", "Shingles"),
    ("Intense itching, skin rash, skin bumps, vesicles, skin fissures, crust formation", "Scabies"),
    ("Fatigue, weight gain, cold hands and feet, mood swings, puffy face, dry skin, hair loss", "Hypothyroidism"),
    ("Joint pain, swelling joints, morning stiffness, fatigue, joint warmth, finger swelling", "Rheumatoid Arthritis"),
    ("I've had a fever that hasn't gone away for nearly a week. I don't feel like eating anything, my stomach hurts, I sometimes feel constipated, and other times I have loose stools. I feel weak all day and have frequent headaches.", "Typhoid"),
]


def run_tests():
    print("=" * 75)
    print(f"CLINICAL TEST SUITE: Evaluating {len(CLINICAL_TEST_CASES)} Real-World & Conversational Cases")
    print("=" * 75)

    passed = 0
    total = len(CLINICAL_TEST_CASES)

    for i, (text, expected) in enumerate(CLINICAL_TEST_CASES, 1):
        detected_symptoms = parse_symptoms_from_text(text, symptom_columns)

        if not detected_symptoms:
            print(f"[{i:02d}/{total}] FAIL [NO SYMPTOMS] - Ground Truth: '{expected}'")
            print(f"       Input: \"{text}\"\n")
            continue

        feat_vec = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)
        for s in detected_symptoms:
            if s in feat_vec.columns:
                feat_vec[s] = 1

        feat_sc = scaler.transform(feat_vec)
        probs = model.predict_proba(feat_sc)[0]
        top_5_idx = np.argsort(probs)[-5:][::-1]
        top_diseases = encoder.inverse_transform(top_5_idx)
        top_probs = probs[top_5_idx]

        top_1 = top_diseases[0]
        top_1_prob = top_probs[0] * 100

        is_correct = (top_1.lower().strip() == expected.lower().strip())

        if is_correct:
            passed += 1
            print(f"[{i:02d}/{total}] PASS - Ground Truth: '{expected}' | Top-1: '{top_1}' ({top_1_prob:.1f}%)")
        else:
            top_3_str = ", ".join([f"{d} ({p*100:.1f}%)" for d, p in zip(top_diseases[:3], top_probs[:3])])
            print(f"[{i:02d}/{total}] FAIL - Ground Truth: '{expected}' | Got Top-1: '{top_1}' ({top_1_prob:.1f}%)")
            print(f"       Detected ({len(detected_symptoms)}): {[symptom_display_name(s) for s in detected_symptoms]}")
            print(f"       Top 3 Predictions: {top_3_str}\n")

    accuracy_pct = (passed / total) * 100
    print("=" * 75)
    print(f"TEST RESULTS SUMMARY: {passed} / {total} Passed ({accuracy_pct:.1f}% Accuracy)")
    print("=" * 75)
    return passed, total

if __name__ == "__main__":
    run_tests()
