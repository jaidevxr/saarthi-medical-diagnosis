"""
Helper utilities: Precise NLP symptom parser, prediction history management,
and general-purpose functions.
"""

import os
import re
import json
from datetime import datetime

# ─────────────────────────────────────────────────
# Prediction History (JSON-based local storage)
# ─────────────────────────────────────────────────
_HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "prediction_history.json")


def _ensure_history_dir():
    os.makedirs(_HISTORY_DIR, exist_ok=True)


def load_prediction_history():
    """Load prediction history from JSON file."""
    _ensure_history_dir()
    if os.path.exists(_HISTORY_FILE):
        try:
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_prediction(symptoms, predictions):
    """
    Append a prediction record to history.
    """
    _ensure_history_dir()
    history = load_prediction_history()
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms": symptoms,
        "predictions": predictions,
    }
    history.insert(0, record)
    history = history[:500]
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def clear_prediction_history():
    """Delete the prediction history file."""
    if os.path.exists(_HISTORY_FILE):
        os.remove(_HISTORY_FILE)


# ─────────────────────────────────────────────────
# Precise NLP Symptom Parser
# ─────────────────────────────────────────────────

_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "also", "since",
    "days", "day", "week", "weeks", "month", "months", "ago", "last",
    "feel", "feeling", "experiencing", "experience", "lot", "really",
    "quite", "getting", "got", "get", "having", "three", "two", "four",
    "five", "six", "seven", "several", "many", "much", "little", "bit",
    "sometimes", "often", "always", "still", "already", "recently",
    "started", "start", "seem", "seems", "like", "think", "going",
    "went", "lately", "past", "suffering",
}

_GENERIC_WORDS = {
    "pain", "skin", "dry", "swollen", "red", "dark", "yellow", "loss",
    "patch", "patches", "discharge", "pressure", "stiff", "stiffness",
    "weakness", "cramps", "fullness", "sore", "dents", "spots", "burning",
    "cough", "fever", "cramping", "warmth", "redness", "dusting", "peeling",
    "flaking", "cracking", "scaling", "thickening", "lesions", "fissures",
    "ulcers", "bumps", "water", "watering", "cold", "hot", "mild", "high",
    "severe", "intense", "sharp", "dull", "persistent",
}

_KEYWORD_MAP = {
    # ── Multi-word Exact Clinical Phrases (High Priority) ──
    "receiving unsterile injections": "receiving_unsterile_injections",
    "receiving unsterile injection": "receiving_unsterile_injections",
    "unsterile injections": "receiving_unsterile_injections",
    "unsterile injection": "receiving_unsterile_injections",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "blood transfusion": "receiving_blood_transfusion",
    "right lower abdominal pain": "right_lower_abdominal_pain",
    "right upper abdominal pain": "right_upper_abdominal_pain",
    "left lower abdominal pain": "left_lower_abdominal_pain",
    "pain behind the eyes": "pain_behind_the_eyes",
    "pain behind eyes": "pain_behind_the_eyes",
    "one sided severe headache": "one_sided_headache",
    "one sided headache": "one_sided_headache",
    "aura before headache": "aura_before_headache",
    "sensitivity to light": "sensitivity_to_light",
    "sensitivity to sound": "sensitivity_to_sound",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "short of breath": "breathlessness",
    "rust colored phlegm": "rusty_sputum",
    "rusty sputum": "rusty_sputum",
    "blood in sputum": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "night sweats": "night_sweats",
    "chronic fatigue": "chronic_fatigue",
    "coughing at night": "coughing_at_night",
    "barking cough": "barking_cough",
    "burning micturition": "burning_micturition",
    "burning urination": "burning_micturition",
    "bladder discomfort": "bladder_discomfort",
    "urinary urgency": "urinary_urgency",
    "foul smell of urine": "foul_smell_of_urine",
    "smelly urine": "foul_smell_of_urine",
    "cloudy urine": "cloudy_urine",
    "yellowish skin": "yellowish_skin",
    "yellow skin": "yellowish_skin",
    "yellowing of eyes": "yellowing_of_eyes",
    "yellow eyes": "yellowing_of_eyes",
    "dark urine": "dark_urine",
    "red spots over body": "red_spots_over_body",
    "red spots": "red_spots_over_body",
    "silver like dusting": "silver_like_dusting",
    "silver dusting": "silver_like_dusting",
    "small dents in nails": "small_dents_in_nails",
    "nail dents": "small_dents_in_nails",
    "inflammatory nails": "inflammatory_nails",
    "joint pain": "joint_pain",
    "joint redness": "joint_redness",
    "joint warmth": "joint_warmth",
    "swelling joints": "swelling_joints",
    "swollen joints": "swelling_joints",
    "foot pain": "foot_pain",
    "knee pain": "knee_pain",
    "hip joint pain": "hip_joint_pain",
    "hip pain": "hip_joint_pain",
    "ear pain": "ear_pain",
    "ear discharge": "ear_discharge",
    "hearing loss": "hearing_loss",
    "ear fullness": "ear_fullness",
    "balance problems": "balance_problems",
    "red eye": "red_eye",
    "redness of eyes": "redness_of_eyes",
    "eye discharge": "eye_discharge",
    "eye itching": "eye_itching",
    "watering from eyes": "watering_from_eyes",
    "crusty eyelids": "crusty_eyelids",
    "photophobia": "photophobia",
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "cold hands and feet": "cold_hands_and_feets",
    "cold hands and feets": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "abdominal pain": "abdominal_pain",
    "stomach pain": "stomach_pain",
    "belly pain": "belly_pain",
    "abdominal cramping": "abdominal_cramping",
    "abdominal bloating": "abdominal_bloating",
    "facial pain": "facial_pain",
    "facial pressure": "facial_pressure",
    "nasal discharge": "nasal_discharge",
    "sinus pressure": "sinus_pressure",
    "post nasal drip": "post_nasal_drip",
    "sore throat": "sore_throat",
    "painful swallowing": "painful_swallowing",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "lymph nodes": "swelled_lymph_nodes",
    "patches in throat": "patches_in_throat",
    "white patches on tongue": "white_patches_on_tongue",
    "white patches on throat": "white_patches_on_tongue",
    "skin tenderness": "skin_tenderness",
    "skin rash": "skin_rash",
    "skin rash blister": "blister",
    "intense itching": "itching",
    "skin bumps": "skin_bumps",
    "skin fissures": "skin_fissures",
    "crust formation": "crust_formation",
    "skin flaking": "skin_flaking",
    "skin cracking": "skin_cracking",
    "skin peeling": "skin_peeling",
    "skin scaling": "skin_scaling",
    "dry skin": "dry_skin",
    "skin burning": "skin_burning",
    "weight gain": "weight_gain",
    "weight loss": "weight_loss",
    "mood swings": "mood_swings",
    "puffy face and eyes": "puffy_face_and_eyes",
    "puffy face": "puffy_face_and_eyes",
    "hair loss": "hair_loss",
    "thinning hair": "hair_loss",
    "morning stiffness": "morning_stiffness",
    "finger swelling": "finger_swelling",
    "back pain": "back_pain",
    "lower back pain": "lower_back_pain",
    "neck pain": "neck_pain",
    "stiff neck": "stiff_neck",
    "chest pain": "chest_pain",
    "chest pressure": "chest_pressure",
    "chest tightness": "chest_tightness",
    "muscle pain": "muscle_pain",
    "body aches": "body_aches",
    "body ache": "body_aches",
    "muscle weakness": "muscle_weakness",
    "weakness in limbs": "weakness_in_limbs",
    "fast heart rate": "fast_heart_rate",
    "rapid heartbeat": "fast_heart_rate",
    "excessive thirst": "excessive_thirst",
    "frequent urination": "polyuria",
    "excessive hunger": "excessive_hunger",
    "blurred vision": "blurred_and_distorted_vision",
    "visual disturbances": "visual_disturbances",
    "visual disturbance": "visual_disturbances",
    "toxic look": "toxic_look_(typhos)",
    "toxic look typhos": "toxic_look_(typhos)",

    # ── Single-word Direct Clinical Symptoms ──
    "itching": "itching",
    "itch": "itching",
    "itchy": "itching",
    "shivering": "shivering",
    "shiver": "shivering",
    "chills": "chills",
    "chill": "chills",
    "acidity": "acidity",
    "heartburn": "acidity",
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "fatigue": "fatigue",
    "anxiety": "anxiety",
    "restlessness": "restlessness",
    "lethargy": "lethargy",
    "cough": "cough",
    "coughing": "cough",
    "high fever": "high_fever",
    "mild fever": "mild_fever",
    "sweating": "sweating",
    "sweat": "sweating",
    "dehydration": "dehydration",
    "indigestion": "indigestion",
    "headache": "headache",
    "migraine": "headache",
    "nausea": "nausea",
    "nauseous": "nausea",
    "constipation": "constipation",
    "diarrhoea": "diarrhoea",
    "diarrhea": "diarrhoea",
    "malaise": "malaise",
    "phlegm": "phlegm",
    "congestion": "congestion",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "cramps": "cramps",
    "cramping": "cramps",
    "bruising": "bruising",
    "obesity": "obesity",
    "slurred speech": "slurred_speech",
    "vertigo": "spinning_movements",
    "depression": "depression",
    "irritability": "irritability",
    "blister": "blister",
    "blisters": "blister",
    "wheezing": "wheezing",
    "stridor": "stridor",
    "hoarseness": "hoarseness",
    "nosebleeds": "nosebleeds",
    "snoring": "snoring",
    "drooling": "drooling",
    "seizures": "seizures",
    "tremors": "tremors",
    "hives": "hives",
    "vesicles": "vesicles",
    "palpitations": "palpitations",
    "blackheads": "blackheads",
    "scurring": "scurring",
    "gout": "joint_pain",
    "tinnitus": "tinnitus_ringing",
    "photosensitivity": "photosensitivity",
}


def parse_symptoms_from_text(text, valid_symptoms):
    """
    Parse free-text user input and map to dataset symptom columns using
    strict phrase and non-generic keyword matching.
    """
    if not text or not text.strip():
        return []

    text_lower = text.lower()
    text_clean = re.sub(r"[^\w\s]", " ", text_lower)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    matched = set()
    remaining_text = text_clean

    # Step 1: Match multi-word phrases first (sorted longest to shortest)
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)

    for kw in sorted_keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, remaining_text):
            symptom = _KEYWORD_MAP[kw]
            if symptom in valid_symptoms:
                matched.add(symptom)
            remaining_text = re.sub(pattern, " ", remaining_text)

    # Step 2: Check remaining text tokens for exact single-word matches
    tokens = remaining_text.split()
    for token in tokens:
        if token in _STOP_WORDS or token in _GENERIC_WORDS or len(token) < 3:
            continue
        if token in valid_symptoms:
            matched.add(token)
        elif token in _KEYWORD_MAP:
            sym = _KEYWORD_MAP[token]
            if sym in valid_symptoms:
                matched.add(sym)

    return sorted(matched)
