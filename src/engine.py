"""
Production Multi-Turn Diagnosis Engine (src/engine.py)
=====================================================
Integrates the verified multi-turn weighted overlap specificity matcher,
information-theoretic entropy question selector, and plain-English question formatter.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
sys.path.insert(0, PROJECT_ROOT)

from src.normalization import SymptomNormalizer

# Load Base Dataset for Canonical Disease Profiles
V1_PATH = os.path.join(DATA_DIR, "training_data_v1.csv")
v1_df = pd.read_csv(V1_PATH)
symptom_cols = [c for c in v1_df.columns if c != "prognosis"]
canonical_diseases = v1_df["prognosis"].unique()
canonical_lower_map = {d.lower().strip(): d for d in canonical_diseases}

alias_map = {
    "asthma": "bronchial asthma",
    "dimorphic hemorrhoids(piles)": "dimorphic hemmorhoids(piles)",
    "peptic ulcer disease": "peptic ulcer diseae"
}

def get_canonical_target(raw_target):
    t_clean = raw_target.lower().strip()
    if t_clean in alias_map:
        t_clean = alias_map[t_clean]
    return canonical_lower_map.get(t_clean, raw_target)


# Build Profiles & Specificity Weights
n_diseases = len(canonical_diseases)
profiles = {}
disease_presence = defaultdict(int)

for d in canonical_diseases:
    sub = v1_df[v1_df["prognosis"] == d]
    active = set(col for col in symptom_cols if (sub[col] > 0).any())
    profiles[d] = active
    for col in active:
        disease_presence[col] += 1

specificity_weights = {}
for col in symptom_cols:
    count = disease_presence.get(col, 0)
    weight = math.log((n_diseases + 1.0) / (count + 1.0)) + 1.0 if count > 0 else 1.0
    specificity_weights[col] = weight

# Plain English Question Template Mapping
QUESTION_TEMPLATES = {
    "burning_micturition": "Are you experiencing a burning or painful sensation when urinating?",
    "polyuria": "Have you noticed an unusually frequent urge to urinate throughout the day or night?",
    "cloudy_urine": "Is your urine appearing unusually cloudy or murky?",
    "dark_urine": "Is your urine darker than usual, resembling tea or cola?",
    "yellowish_skin": "Have you noticed any yellowing of your skin or eyes (jaundice)?",
    "yellowing_of_eyes": "Are the whites of your eyes appearing yellowish?",
    "chest_pain": "Are you feeling chest tightness, pressure, or sharp chest pain?",
    "breathlessness": "Do you feel short of breath or have trouble breathing deeply?",
    "cough": "Do you have a persistent cough?",
    "phlegm": "Are you coughing up thick mucus or phlegm?",
    "high_fever": "Do you have a high body temperature or fever above 100°F (38°C)?",
    "chills": "Are you feeling sudden cold chills or shivering uncontrollably?",
    "sweating": "Have you experienced excessive sweating or drenching night sweats?",
    "itching": "Are you experiencing persistent skin itching or irritation?",
    "skin_rash": "Do you have any visible red skin rashes, bumps, or patches?",
    "joint_pain": "Do you have aching, pain, or stiffness in your joints?",
    "swelling_joints": "Are any of your joints visibly swollen, warm, or red?",
    "stiffness_in_joints": "Do your joints feel stiff, particularly in the morning?",
    "stiff_neck": "Is your neck stiff or difficult to bend forward comfortably?",
    "vomiting": "Have you vomited or felt severe nausea?",
    "diarrhoea": "Are you experiencing loose or watery bowel movements (diarrhea)?",
    "abdominal_pain": "Do you have pain or cramping in your stomach or abdomen?",
    "stomach_pain": "Do you have pain in your stomach area?",
    "acidity": "Do you experience heartburn, acid reflux, or a sour taste in your throat?",
    "headache": "Do you have a severe or throbbing headache?",
    "dizziness": "Do you feel lightheaded, faint, or unsteadily dizzy?",
    "spinning_movements": "Do you feel like the room or surroundings are spinning around you (vertigo)?",
    "loss_of_balance": "Are you having difficulty keeping your balance while standing or walking?",
    "fatigue": "Are you feeling unusually exhausted or devoid of energy?",
    "unexplained_weight_loss": "Have you experienced noticeable weight loss without trying?",
    "increased_appetite": "Are you feeling constantly thirsty or excessively hungry?",
    "irregular_sugar_level": "Do you have a history of high blood sugar or diabetes?",
    "blood_pressure_high": "Do you have elevated blood pressure readings?",
    "pus_filled_pimples": "Do you have pus-filled pimples or severe breakouts on your face or back?",
    "skin_peeling": "Is your skin flaking, scaling, or peeling off in patches?"
}

def format_symptom_question(col_name):
    if col_name in QUESTION_TEMPLATES:
        return QUESTION_TEMPLATES[col_name]
    clean_name = col_name.replace("_", " ")
    return f"Are you experiencing {clean_name}?"


class MultiTurnDiagnosisEngine:
    def __init__(self):
        self.normalizer = SymptomNormalizer()
        self.symptom_cols = symptom_cols
        self.canonical_diseases = canonical_diseases

    def compute_scores_and_candidates(self, present_syms, absent_syms):
        scores = {}
        for d in self.canonical_diseases:
            prof = profiles[d]

            if any(abs_s in prof for abs_s in absent_syms):
                scores[d] = 0.0
                continue

            inter = set(present_syms).intersection(prof)
            if not inter:
                scores[d] = 0.0
                continue

            w_inter = sum(specificity_weights[s] for s in inter)
            w_query = sum(specificity_weights[s] for s in present_syms) if present_syms else 1.0
            w_prof = sum(specificity_weights[s] for s in prof)

            cont = w_inter / w_query if w_query > 0 else 0.0
            rec = w_inter / w_prof if w_prof > 0 else 0.0
            scores[d] = 0.70 * cont + 0.30 * rec

        max_score = max(scores.values()) if scores else 0.0
        if max_score == 0.0:
            return scores, [], "Uncertain / Additional Info Required", []

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [pair[0] for pair in sorted_scores if (max_score - pair[1]) <= 0.05]
        top1 = sorted_scores[0][0]
        top3 = [pair[0] for pair in sorted_scores[:3]]

        return scores, candidates, top1, top3

    def select_optimal_clarifying_question(self, candidates, asked_syms):
        if len(candidates) <= 1:
            return None

        candidate_profiles = [profiles[d] for d in candidates]
        all_cand_syms = set()
        for prof in candidate_profiles:
            all_cand_syms.update(prof)

        candidate_questions = [s for s in all_cand_syms if s not in asked_syms]

        if not candidate_questions:
            return None

        best_symptom = None
        best_split_score = -1.0
        n_cand = len(candidates)

        for sym in candidate_questions:
            c_yes = sum(1 for prof in candidate_profiles if sym in prof)
            c_no = n_cand - c_yes

            if c_yes == 0 or c_no == 0:
                continue

            p_yes = c_yes / n_cand
            p_no = c_no / n_cand
            entropy = - (p_yes * math.log2(p_yes) + p_no * math.log2(p_no))

            w_sym = specificity_weights.get(sym, 1.0)
            score = entropy * w_sym

            if score > best_split_score:
                best_split_score = score
                best_symptom = sym

        return best_symptom

    def predict_initial(self, text):
        parsed = self.normalizer.extract_symptoms(text, self.symptom_cols)
        matched = parsed["matched"]
        unmatched = parsed["unmatched"]

        present_syms = set(matched)
        absent_syms = set()
        asked_syms = set(matched)

        scores, candidates, top1, top3 = self.compute_scores_and_candidates(present_syms, absent_syms)

        if not matched or max(scores.values(), default=0.0) < 0.20:
            return {
                "initial_text": text,
                "matched_symptoms": matched,
                "unmatched_tokens": unmatched,
                "present_symptoms": list(present_syms),
                "absent_symptoms": list(absent_syms),
                "asked_symptoms": list(asked_syms),
                "rounds_asked": 0,
                "candidate_diseases": [],
                "primary_diagnosis": "Uncertain / Additional Info Required",
                "top3_differential": [],
                "next_question_symptom": None,
                "next_question_text": None,
                "is_resolved": True
            }

        next_symptom = self.select_optimal_clarifying_question(candidates, asked_syms)
        is_resolved = len(candidates) <= 1 or next_symptom is None

        return {
            "initial_text": text,
            "matched_symptoms": matched,
            "unmatched_tokens": unmatched,
            "present_symptoms": list(present_syms),
            "absent_symptoms": list(absent_syms),
            "asked_symptoms": list(asked_syms),
            "rounds_asked": 0,
            "candidate_diseases": candidates,
            "primary_diagnosis": top1,
            "confidence": float(max(scores.values(), default=0.0) * 100),
            "explanation": f"Primary diagnosis is {top1} with {len(candidates)} candidate disease(s) remaining.",
            "top3_differential": top3,
            "next_question_symptom": next_symptom,
            "next_question_text": format_symptom_question(next_symptom) if next_symptom else None,
            "is_resolved": is_resolved
        }

    def process_followup(self, present_symptoms, absent_symptoms, asked_symptoms, user_answer_yes_no, current_question_symptom, rounds_asked=0):
        present_set = set(present_symptoms or [])
        absent_set = set(absent_symptoms or [])
        asked_set = set(asked_symptoms or [])

        if current_question_symptom:
            asked_set.add(current_question_symptom)
            if user_answer_yes_no.upper() in ["YES", "Y", "TRUE", "1"]:
                present_set.add(current_question_symptom)
            else:
                absent_set.add(current_question_symptom)

        scores, candidates, top1, top3 = self.compute_scores_and_candidates(present_set, absent_set)
        rounds_asked += 1

        next_symptom = None
        if rounds_asked < 2:
            next_symptom = self.select_optimal_clarifying_question(candidates, asked_set)

        is_resolved = len(candidates) <= 1 or rounds_asked >= 2 or next_symptom is None

        return {
            "present_symptoms": list(present_set),
            "absent_symptoms": list(absent_set),
            "asked_symptoms": list(asked_set),
            "rounds_asked": rounds_asked,
            "candidate_diseases": candidates,
            "primary_diagnosis": top1,
            "top3_differential": top3,
            "next_question_symptom": next_symptom,
            "next_question_text": format_symptom_question(next_symptom) if next_symptom else None,
            "is_resolved": is_resolved
        }
