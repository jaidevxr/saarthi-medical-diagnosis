"""
Helper utilities: NLP symptom parser, prediction history management,
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

    Parameters
    ----------
    symptoms : list[str]
        Human-readable symptom names selected by the user.
    predictions : list[dict]
        Each dict has keys 'disease' and 'probability'.
    """
    _ensure_history_dir()
    history = load_prediction_history()
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms": symptoms,
        "predictions": predictions,
    }
    history.insert(0, record)  # newest first
    # Keep last 500 entries
    history = history[:500]
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def clear_prediction_history():
    """Delete the prediction history file."""
    if os.path.exists(_HISTORY_FILE):
        os.remove(_HISTORY_FILE)


# ─────────────────────────────────────────────────
# Simple NLP Symptom Parser (NO external libraries)
# ─────────────────────────────────────────────────

# Common English stop words (subset)
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

# Extended keyword mapping: user might type any of these → mapped symptom
_KEYWORD_MAP = {
    # ── Direct symptom keywords ──
    "itch": "itching",
    "itchy": "itching",
    "itching": "itching",
    "rash": "skin_rash",
    "skin rash": "skin_rash",
    "eruptions": "nodal_skin_eruptions",
    "sneezing": "continuous_sneezing",
    "sneeze": "continuous_sneezing",
    "shiver": "shivering",
    "shivering": "shivering",
    "chill": "chills",
    "chills": "chills",
    "joint pain": "joint_pain",
    "joint ache": "joint_pain",
    "stomach pain": "stomach_pain",
    "stomach ache": "stomach_pain",
    "tummy pain": "stomach_pain",
    "acidity": "acidity",
    "acid reflux": "acidity",
    "heartburn": "acidity",
    "ulcer": "ulcers_on_tongue",
    "tongue ulcer": "ulcers_on_tongue",
    "mouth ulcer": "ulcers_on_tongue",
    "muscle wasting": "muscle_wasting",
    "vomit": "vomiting",
    "vomiting": "vomiting",
    "throw up": "vomiting",
    "throwing up": "vomiting",
    "puke": "vomiting",
    "burning urination": "burning_micturition",
    "burning micturition": "burning_micturition",
    "burning pee": "burning_micturition",
    "painful urination": "burning_micturition",
    "spotting": "spotting_urination",
    "fatigue": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "exhausted": "fatigue",
    "weight gain": "weight_gain",
    "gaining weight": "weight_gain",
    "anxiety": "anxiety",
    "anxious": "anxiety",
    "nervous": "anxiety",
    "cold hands": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "cold extremities": "cold_hands_and_feets",
    "mood swings": "mood_swings",
    "mood swing": "mood_swings",
    "weight loss": "weight_loss",
    "losing weight": "weight_loss",
    "restless": "restlessness",
    "restlessness": "restlessness",
    "lethargy": "lethargy",
    "lethargic": "lethargy",
    "sluggish": "lethargy",
    "patches throat": "patches_in_throat",
    "throat patches": "patches_in_throat",
    "irregular sugar": "irregular_sugar_level",
    "blood sugar": "irregular_sugar_level",
    "sugar level": "irregular_sugar_level",
    "cough": "cough",
    "coughing": "cough",
    "fever": "high_fever",
    "high fever": "high_fever",
    "temperature": "high_fever",
    "sunken eyes": "sunken_eyes",
    "breathless": "breathlessness",
    "breathlessness": "breathlessness",
    "shortness of breath": "breathlessness",
    "short of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "sweating": "sweating",
    "sweat": "sweating",
    "perspiration": "sweating",
    "dehydration": "dehydration",
    "dehydrated": "dehydration",
    "indigestion": "indigestion",
    "headache": "headache",
    "head ache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "yellow skin": "yellowish_skin",
    "yellowish skin": "yellowish_skin",
    "jaundice": "yellowish_skin",
    "dark urine": "dark_urine",
    "nausea": "nausea",
    "nauseous": "nausea",
    "queasy": "nausea",
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    "appetite loss": "loss_of_appetite",
    "eye pain": "pain_behind_the_eyes",
    "pain behind eyes": "pain_behind_the_eyes",
    "back pain": "back_pain",
    "backache": "back_pain",
    "lower back pain": "back_pain",
    "constipation": "constipation",
    "constipated": "constipation",
    "abdominal pain": "abdominal_pain",
    "abdomen pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose stools": "diarrhoea",
    "loose motions": "diarrhoea",
    "mild fever": "mild_fever",
    "low grade fever": "mild_fever",
    "slight fever": "mild_fever",
    "yellow urine": "yellow_urine",
    "yellow eyes": "yellowing_of_eyes",
    "yellowing eyes": "yellowing_of_eyes",
    "liver failure": "acute_liver_failure",
    "fluid overload": "fluid_overload",
    "swollen stomach": "swelling_of_stomach",
    "stomach swelling": "swelling_of_stomach",
    "swollen lymph": "swelled_lymph_nodes",
    "lymph nodes": "swelled_lymph_nodes",
    "malaise": "malaise",
    "general discomfort": "malaise",
    "unwell": "malaise",
    "blurred vision": "blurred_and_distorted_vision",
    "blurry vision": "blurred_and_distorted_vision",
    "distorted vision": "blurred_and_distorted_vision",
    "phlegm": "phlegm",
    "mucus": "phlegm",
    "throat irritation": "throat_irritation",
    "sore throat": "throat_irritation",
    "scratchy throat": "throat_irritation",
    "red eyes": "redness_of_eyes",
    "eye redness": "redness_of_eyes",
    "sinus pressure": "sinus_pressure",
    "sinus": "sinus_pressure",
    "runny nose": "runny_nose",
    "running nose": "runny_nose",
    "congestion": "congestion",
    "nasal congestion": "congestion",
    "stuffy nose": "congestion",
    "blocked nose": "congestion",
    "chest pain": "chest_pain",
    "chest tightness": "chest_pain",
    "weak limbs": "weakness_in_limbs",
    "limb weakness": "weakness_in_limbs",
    "fast heart rate": "fast_heart_rate",
    "rapid heartbeat": "fast_heart_rate",
    "heart racing": "fast_heart_rate",
    "tachycardia": "fast_heart_rate",
    "bowel pain": "pain_during_bowel_movements",
    "anal pain": "pain_in_anal_region",
    "bloody stool": "bloody_stool",
    "blood in stool": "bloody_stool",
    "neck pain": "neck_pain",
    "stiff neck": "stiff_neck",
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "cramps": "cramps",
    "cramping": "cramps",
    "bruising": "bruising",
    "bruise": "bruising",
    "obesity": "obesity",
    "overweight": "obesity",
    "swollen legs": "swollen_legs",
    "leg swelling": "swollen_legs",
    "varicose": "swollen_blood_vessels",
    "puffy face": "puffy_face_and_eyes",
    "puffy eyes": "puffy_face_and_eyes",
    "thyroid": "enlarged_thyroid",
    "brittle nails": "brittle_nails",
    "nail brittle": "brittle_nails",
    "excessive hunger": "excessive_hunger",
    "always hungry": "excessive_hunger",
    "tingling lips": "drying_and_tingling_lips",
    "slurred speech": "slurred_speech",
    "speech difficulty": "slurred_speech",
    "knee pain": "knee_pain",
    "hip pain": "hip_joint_pain",
    "hip joint pain": "hip_joint_pain",
    "muscle weakness": "muscle_weakness",
    "weak muscles": "muscle_weakness",
    "swollen joints": "swelling_joints",
    "joint swelling": "swelling_joints",
    "stiffness": "movement_stiffness",
    "movement stiffness": "movement_stiffness",
    "spinning": "spinning_movements",
    "vertigo": "spinning_movements",
    "loss of balance": "loss_of_balance",
    "imbalance": "loss_of_balance",
    "unsteady": "unsteadiness",
    "unsteadiness": "unsteadiness",
    "one side weakness": "weakness_of_one_body_side",
    "loss of smell": "loss_of_smell",
    "anosmia": "loss_of_smell",
    "cant smell": "loss_of_smell",
    "bladder discomfort": "bladder_discomfort",
    "bladder pain": "bladder_discomfort",
    "foul urine": "foul_smell_of_urine",
    "smelly urine": "foul_smell_of_urine",
    "urge to urinate": "continuous_feel_of_urine",
    "frequent urination": "continuous_feel_of_urine",
    "gas": "passage_of_gases",
    "flatulence": "passage_of_gases",
    "bloating": "passage_of_gases",
    "internal itch": "internal_itching",
    "internal itching": "internal_itching",
    "depression": "depression",
    "depressed": "depression",
    "sad": "depression",
    "irritable": "irritability",
    "irritability": "irritability",
    "muscle pain": "muscle_pain",
    "body pain": "muscle_pain",
    "body ache": "muscle_pain",
    "myalgia": "muscle_pain",
    "confusion": "altered_sensorium",
    "altered sensorium": "altered_sensorium",
    "disoriented": "altered_sensorium",
    "red spots": "red_spots_over_body",
    "red patches": "red_spots_over_body",
    "menstrual irregularity": "abnormal_menstruation",
    "irregular periods": "abnormal_menstruation",
    "abnormal menstruation": "abnormal_menstruation",
    "skin patches": "dischromic_patches",
    "discoloration": "dischromic_patches",
    "watery eyes": "watering_from_eyes",
    "tearing": "watering_from_eyes",
    "increased appetite": "increased_appetite",
    "polyuria": "polyuria",
    "excessive urination": "polyuria",
    "family history": "family_history",
    "genetic": "family_history",
    "hereditary": "family_history",
    "sputum": "mucoid_sputum",
    "rusty sputum": "rusty_sputum",
    "cannot concentrate": "lack_of_concentration",
    "concentration": "lack_of_concentration",
    "focus": "lack_of_concentration",
    "visual disturbance": "visual_disturbances",
    "vision problems": "visual_disturbances",
    "blood transfusion": "receiving_blood_transfusion",
    "unsterile injection": "receiving_unsterile_injections",
    "coma": "coma",
    "unconscious": "coma",
    "stomach bleeding": "stomach_bleeding",
    "distended abdomen": "distention_of_abdomen",
    "alcohol": "history_of_alcohol_consumption",
    "drinking": "history_of_alcohol_consumption",
    "blood sputum": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "calf veins": "prominent_veins_on_calf",
    "palpitations": "palpitations",
    "heart pounding": "palpitations",
    "painful walking": "painful_walking",
    "walk pain": "painful_walking",
    "pimples": "pus_filled_pimples",
    "acne": "pus_filled_pimples",
    "blackheads": "blackheads",
    "scarring": "scurring",
    "skin peeling": "skin_peeling",
    "peeling skin": "skin_peeling",
    "silver dusting": "silver_like_dusting",
    "nail dents": "small_dents_in_nails",
    "inflammatory nails": "inflammatory_nails",
    "nail inflammation": "inflammatory_nails",
    "blister": "blister",
    "blisters": "blister",
    "red sore nose": "red_sore_around_nose",
    "nose sore": "red_sore_around_nose",
    "yellow crust": "yellow_crust_ooze",
    "oozing": "yellow_crust_ooze",
    "wheezing": "breathlessness",
    "ear pain": "headache",
}


def parse_symptoms_from_text(text, valid_symptoms):
    """
    Parse free-text user input and map to dataset symptom columns.

    Pipeline: lowercase -> remove punctuation -> tokenize
              -> remove stop words -> keyword matching

    Parameters
    ----------
    text : str
        User's natural language description, e.g.
        "I have fever, cough and headache for three days."
    valid_symptoms : list[str]
        The 132 symptom column names from the dataset.

    Returns
    -------
    list[str]
        Matched symptom column names.
    """
    if not text or not text.strip():
        return []

    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove punctuation (keep spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Step 3: normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    matched = set()

    # Step 4: Try matching multi-word keyword phrases first (longer first)
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)
    remaining_text = text
    for kw in sorted_keywords:
        if kw in remaining_text:
            symptom = _KEYWORD_MAP[kw]
            if symptom in valid_symptoms:
                matched.add(symptom)
            # Remove matched phrase to avoid double-counting
            remaining_text = remaining_text.replace(kw, " ", 1)

    # Step 5: Tokenise remaining text and try individual words
    tokens = remaining_text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    for token in tokens:
        # Direct column name match
        if token in valid_symptoms:
            matched.add(token)
        # Match against column names with underscores removed
        for sym in valid_symptoms:
            clean = sym.replace("_", "")
            if token == clean or token in sym.split("_"):
                matched.add(sym)

    return sorted(matched)
