"""
User 20 Web Clinical Scenarios Test Suite
Evaluates natural conversational text for 20 real-world medical cases.
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

USER_TEST_CASES = [
    (
        "1. Dengue",
        "I've had a very high fever for the last four days along with severe body pain and pain behind my eyes. Every joint hurts, I have a terrible headache, and today I noticed a reddish rash on my arms. I've also been feeling nauseous and have no appetite.",
        ["Dengue", "Chikungunya", "Malaria", "Viral Fever", "Typhoid"]
    ),
    (
        "2. Malaria",
        "Every evening I suddenly develop chills and start shivering uncontrollably. After an hour I get a high fever, and once it goes away I sweat heavily. This has been happening repeatedly for the past five days. I also have headaches and feel extremely weak.",
        ["Malaria", "Dengue", "Typhoid", "Viral Fever"]
    ),
    (
        "3. Typhoid",
        "I've had a fever that hasn't gone away for nearly a week. I don't feel like eating anything, my stomach hurts, I sometimes feel constipated, and other times I have loose stools. I feel weak all day and have frequent headaches.",
        ["Typhoid", "Gastroenteritis", "Food Poisoning", "Viral Fever"]
    ),
    (
        "4. Sinusitis",
        "For over a week my nose has been blocked and thick yellow mucus keeps coming out. My cheeks and forehead feel heavy, especially when I bend forward. I have a dull headache, my throat feels irritated because mucus keeps dripping down, and I can't smell food properly anymore.",
        ["Sinusitis", "Allergic Rhinitis", "Common Cold", "Upper Respiratory Infection"]
    ),
    (
        "5. Pneumonia",
        "I've had a fever with chills for several days and a cough producing thick yellow sputum. Taking deep breaths causes chest pain, and walking even a short distance makes me feel short of breath. I feel exhausted all the time.",
        ["Pneumonia", "Acute Bronchitis", "COPD", "Tuberculosis"]
    ),
    (
        "6. Kidney Stone",
        "I suddenly developed unbearable pain in my lower back that comes and goes in waves. The pain moves toward my groin. Urinating burns, I noticed a little blood in my urine, and I feel nauseous because of the pain.",
        ["Kidney Stones", "Urinary tract infection", "Pyelonephritis"]
    ),
    (
        "7. Urinary Tract Infection",
        "For the last two days I need to urinate very frequently, but only a little comes out each time. It burns badly while urinating, my urine has a strong smell, and I have mild pain in my lower abdomen with a slight fever.",
        ["Urinary tract infection", "Bladder Infection", "Kidney Stones"]
    ),
    (
        "8. Migraine",
        "Every few weeks I develop a severe throbbing headache on one side of my head. Bright lights and loud sounds make it unbearable. Sometimes I see flashing lights before the headache starts and I often feel nauseous afterwards.",
        ["Migraine", "Migraine with Aura", "Cluster Headache", "Tension Headache"]
    ),
    (
        "9. Asthma",
        "Whenever I exercise or breathe cold air, I start wheezing and my chest feels tight. I have difficulty breathing and a dry cough that becomes worse at night. Using an inhaler usually helps.",
        ["Bronchial Asthma", "Allergic Rhinitis", "COPD"]
    ),
    (
        "10. Tuberculosis",
        "I've had a cough for more than a month. Recently I've started coughing up small amounts of blood. I wake up sweating at night, have lost weight without trying, and usually get a mild fever every evening.",
        ["Tuberculosis", "Pneumonia", "Acute Bronchitis"]
    ),
    (
        "11. Diabetes",
        "I'm constantly thirsty and drink water all day, yet my mouth still feels dry. I have to urinate many times during the day and even wake up several times at night. Despite eating normally, I've been losing weight and my vision sometimes becomes blurry.",
        ["Diabetes", "Hyperglycemia", "Hypoglycemia"]
    ),
    (
        "12. Hepatitis",
        "My eyes and skin have started looking yellow. My urine has become very dark while my stools are unusually pale. I don't feel like eating, I often feel nauseous, and there's a dull pain below my right ribs.",
        ["Hepatitis B", "hepatitis A", "Jaundice", "Chronic cholestasis"]
    ),
    (
        "13. Appendicitis",
        "It started as a stomach ache around my belly button, but within a day the pain shifted to the lower right side of my abdomen. Walking or coughing makes it much worse. I've also had nausea, vomiting, and a low fever.",
        ["Appendicitis", "Gastroenteritis"]
    ),
    (
        "14. Food Poisoning",
        "A few hours after eating street food I started vomiting repeatedly and developed severe stomach cramps followed by watery diarrhea. I have a mild fever and feel dizzy because I can't keep food or water down.",
        ["Food Poisoning", "Gastroenteritis"]
    ),
    (
        "15. GERD",
        "After eating, I often feel a burning sensation rising from my stomach into my chest. Sour liquid sometimes comes into my mouth, especially when lying down. I frequently burp and have a persistent sore throat in the mornings.",
        ["GERD", "Gastritis", "Peptic ulcer diseae"]
    ),
    (
        "16. Chickenpox",
        "I developed a fever two days ago and now my body is covered with itchy red spots that are turning into fluid-filled blisters. New spots keep appearing every day, and I'm feeling weak with a mild headache.",
        ["Chicken pox", "Shingles", "Impetigo"]
    ),
    (
        "17. Arthritis",
        "For several months my knees and fingers have been painful and stiff, especially when I wake up in the morning. The stiffness lasts for almost an hour before I can move normally. My finger joints sometimes become swollen and warm.",
        ["Rheumatoid Arthritis", "Osteoarthristis", "Psoriatic Arthritis"]
    ),
    (
        "18. Allergy",
        "Every time I'm around dust or during spring I start sneezing continuously. My nose becomes runny, my eyes itch and water, and I sometimes develop an itchy rash on my skin, although I never have a fever.",
        ["Allergy", "Allergic Rhinitis"]
    ),
    (
        "19. COVID-19 / Influenza",
        "I've had a fever, sore throat, dry cough, and body aches for three days. Food has almost no taste, I can't smell anything, and I feel unusually tired. Climbing stairs leaves me slightly short of breath.",
        ["Influenza", "Common Cold", "Pneumonia"]
    ),
    (
        "20. Mixed Viral Case",
        "For nearly a week I've had a low fever, severe tiredness, headaches, muscle pain, loss of appetite, and a blocked nose.",
        ["Common Cold", "Influenza", "Sinusitis", "Viral Fever"]
    ),
]


def run_tests():
    print("=" * 80)
    print(f"USER WEB SCENARIOS TEST SUITE: Evaluating {len(USER_TEST_CASES)} Natural Language Cases")
    print("=" * 80)

    passed = 0
    total = len(USER_TEST_CASES)

    for case_num, text, expected_list in USER_TEST_CASES:
        detected_symptoms = parse_symptoms_from_text(text, symptom_columns)

        if not detected_symptoms:
            print(f"[{case_num}] FAIL [NO SYMPTOMS DETECTED]\n       Input: \"{text}\"\n")
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

        # Check if top_1 is in expected_list or top 3 matches expected
        matched_expected = None
        for exp in expected_list:
            if top_1.lower().strip() == exp.lower().strip():
                matched_expected = exp
                break

        is_top3_match = any(
            d.lower().strip() in [e.lower().strip() for e in expected_list]
            for d in top_diseases[:3]
        )

        if matched_expected:
            passed += 1
            print(f"[{case_num}] PASS (Top-1 Match) — '{top_1}' ({top_1_prob:.1f}%)")
            print(f"       Detected ({len(detected_symptoms)}): {[symptom_display_name(s) for s in detected_symptoms]}")
        elif is_top3_match:
            passed += 1
            top_3_str = ", ".join([f"{d} ({p*100:.1f}%)" for d, p in zip(top_diseases[:3], top_probs[:3])])
            print(f"[{case_num}] PASS (Top-3 Match) — Top 3: {top_3_str}")
            print(f"       Detected ({len(detected_symptoms)}): {[symptom_display_name(s) for s in detected_symptoms]}")
        else:
            top_3_str = ", ".join([f"{d} ({p*100:.1f}%)" for d, p in zip(top_diseases[:3], top_probs[:3])])
            print(f"[{case_num}] FAIL — Got Top-1: '{top_1}' ({top_1_prob:.1f}%) | Expected one of: {expected_list}")
            print(f"       Detected ({len(detected_symptoms)}): {[symptom_display_name(s) for s in detected_symptoms]}")
            print(f"       Top 3 Predictions: {top_3_str}\n")

    accuracy_pct = (passed / total) * 100
    print("=" * 80)
    print(f"TEST RESULTS SUMMARY: {passed} / {total} Passed ({accuracy_pct:.1f}% Accuracy)")
    print("=" * 80)
    return passed, total

if __name__ == "__main__":
    run_tests()
