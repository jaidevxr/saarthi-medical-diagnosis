"""
Helper utilities: Advanced Conversational NLP symptom parser, prediction history,
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
# Advanced Conversational NLP Symptom Parser
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
    "went", "lately", "past", "suffering", "nearly", "all", "day", "ive",
    "hasnt", "dont", "im", "every", "became", "becomes", "keeps", "came",
}

_GENERIC_WORDS = {
    "pain", "skin", "dry", "swollen", "red", "dark", "yellow", "loss",
    "patch", "patches", "discharge", "pressure", "stiff", "stiffness",
    "weakness", "cramps", "fullness", "sore", "dents", "spots", "burning",
    "cough", "cramping", "warmth", "redness", "dusting", "peeling",
    "flaking", "cracking", "scaling", "thickening", "lesions", "fissures",
    "ulcers", "bumps", "water", "watering", "cold", "hot", "mild", "high",
    "severe", "intense", "sharp", "dull", "persistent",
}

# Rich Conversational Phrase Mapping
_KEYWORD_MAP = {
    # ── Conversational & Natural Expressions ──
    "pain behind my eyes": "pain_behind_the_eyes",
    "pain behind the eyes": "pain_behind_the_eyes",
    "pain behind eyes": "pain_behind_the_eyes",
    "every joint hurts": "joint_pain",
    "joint hurts": "joint_pain",
    "joint pain": "joint_pain",
    "reddish rash": "skin_rash",
    "itchy rash": "skin_rash",
    "red spots": "red_spots_over_body",
    "red rash": "skin_rash",
    "fluid filled blisters": "blister",
    "blisters": "blister",
    "blister": "blister",

    "fever that hasn t gone away": "high_fever",
    "fever for nearly a week": "high_fever",
    "fever for a week": "high_fever",
    "fever for days": "high_fever",
    "very high fever": "high_fever",
    "high fever": "high_fever",
    "mild fever": "mild_fever",
    "slight fever": "mild_fever",
    "fever": "high_fever",
    "fevers": "high_fever",

    "don t feel like eating anything": "loss_of_appetite",
    "don t feel like eating": "loss_of_appetite",
    "dont feel like eating": "loss_of_appetite",
    "not feeling like eating": "loss_of_appetite",
    "no desire to eat": "loss_of_appetite",
    "can t eat anything": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "loss of appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",

    "stomach ache around my belly button": "stomach_pain",
    "lower right side of my abdomen": "right_lower_abdominal_pain",
    "lower right side of abdomen": "right_lower_abdominal_pain",
    "right side of my abdomen": "right_lower_abdominal_pain",
    "right lower abdominal pain": "right_lower_abdominal_pain",
    "stomach hurts": "stomach_pain",
    "stomach hurt": "stomach_pain",
    "belly hurts": "belly_pain",
    "tummy hurts": "stomach_pain",
    "stomach ache": "stomach_pain",
    "stomach pain": "stomach_pain",
    "stomach cramps": "cramps",
    "abdominal pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "below my right ribs": "right_upper_abdominal_pain",
    "right ribs": "right_upper_abdominal_pain",

    "feel constipated": "constipation",
    "feeling constipated": "constipation",
    "sometimes constipated": "constipation",
    "constipated": "constipation",
    "constipation": "constipation",

    "loose stools": "diarrhoea",
    "loose stool": "diarrhoea",
    "watery stools": "diarrhoea",
    "watery diarrhea": "diarrhoea",
    "loose motions": "diarrhoea",
    "loose motion": "diarrhoea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "pale stools": "clay_colored_stool",
    "pale stool": "clay_colored_stool",

    "feel weak all day": "fatigue",
    "feel weak": "fatigue",
    "feeling weak": "fatigue",
    "weak all day": "fatigue",
    "extremely weak": "fatigue",
    "unusually tired": "fatigue",
    "feeling tired": "fatigue",
    "exhausted all the time": "fatigue",
    "exhausted": "fatigue",
    "general weakness": "general_weakness",
    "weakness": "fatigue",
    "fatigue": "fatigue",

    "frequent headaches": "headache",
    "frequent headache": "headache",
    "terrible headache": "headache",
    "dull headache": "headache",
    "severe headache": "headache",
    "throbbing headache": "one_sided_headache",
    "one side of my head": "one_sided_headache",
    "headaches": "headache",
    "headache": "headache",

    "nose has been blocked": "congestion",
    "blocked nose": "congestion",
    "thick yellow mucus": "nasal_discharge",
    "yellow mucus": "nasal_discharge",
    "nasal discharge": "nasal_discharge",
    "cheeks and forehead feel heavy": "facial_pressure",
    "forehead feel heavy": "facial_pressure",
    "cheeks feel heavy": "facial_pressure",
    "facial pressure": "facial_pressure",
    "mucus keeps dripping down": "post_nasal_drip",
    "dripping down": "post_nasal_drip",
    "post nasal drip": "post_nasal_drip",
    "can t smell food": "loss_of_smell",
    "can t smell anything": "loss_of_smell",
    "can t smell": "loss_of_smell",
    "no smell": "loss_of_smell",
    "loss of smell": "loss_of_smell",
    "no taste": "loss_of_taste",
    "almost no taste": "loss_of_taste",
    "loss of taste": "loss_of_taste",

    "cough producing thick yellow sputum": "mucoid_sputum",
    "coughing up small amounts of blood": "blood_in_sputum",
    "coughing up blood": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "blood in sputum": "blood_in_sputum",
    "dry cough": "dry_cough",
    "cough at night": "coughing_at_night",
    "worse at night": "coughing_at_night",

    "pain in my lower back": "lower_back_pain",
    "unbearable pain in my lower back": "lower_back_pain",
    "moves toward my groin": "groin_pain",
    "groin": "groin_pain",
    "urinating burns": "burning_micturition",
    "burns badly while urinating": "burning_micturition",
    "burns while urinating": "burning_micturition",
    "burning urination": "burning_micturition",
    "blood in my urine": "blood_in_urine",
    "little blood in my urine": "blood_in_urine",
    "blood in urine": "blood_in_urine",
    "dark urine": "dark_urine",
    "urinate very frequently": "polyuria",
    "urinate many times": "polyuria",
    "urinate several times": "polyuria",
    "frequent urination": "polyuria",
    "strong smell": "foul_smell_of_urine",
    "strong smell of urine": "foul_smell_of_urine",
    "pain in my lower abdomen": "pelvic_pain",
    "lower abdomen": "pelvic_pain",

    "constantly thirsty": "excessive_thirst",
    "thirsty": "excessive_thirst",
    "mouth still feels dry": "dry_mouth",
    "mouth feels dry": "dry_mouth",
    "losing weight": "weight_loss",
    "lost weight": "weight_loss",
    "weight loss": "weight_loss",
    "vision sometimes becomes blurry": "blurred_and_distorted_vision",
    "blurry vision": "blurred_and_distorted_vision",

    "burning sensation rising": "acidity",
    "burning sensation": "acidity",
    "sour liquid": "acid_reflux",
    "burp": "passage_of_gases",
    "burping": "passage_of_gases",

    "wake up in the morning": "morning_stiffness",
    "morning stiffness": "morning_stiffness",
    "swollen and warm": "swelling_joints",
    "swollen joints": "swelling_joints",
    "knee and fingers": "joint_pain",
    "knees and fingers": "joint_pain",

    "sneezing continuously": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "nose becomes runny": "runny_nose",
    "runny nose": "runny_nose",
    "eyes itch and water": "watering_from_eyes",
    "eyes itch": "eye_itching",
    "watering eyes": "watering_from_eyes",

    "flashing lights": "visual_disturbances",
    "bright lights and loud sounds": "sensitivity_to_light",

    "eyes and skin have started looking yellow": "yellowing_of_eyes",
    "yellow eyes": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin",

    # ── Standard Clinical Keywords ──
    "toxic look": "toxic_look_(typhos)",
    "toxic look typhos": "toxic_look_(typhos)",
    "receiving unsterile injections": "receiving_unsterile_injections",
    "receiving unsterile injection": "receiving_unsterile_injections",
    "unsterile injections": "receiving_unsterile_injections",
    "unsterile injection": "receiving_unsterile_injections",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "blood transfusion": "receiving_blood_transfusion",
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "short of breath": "breathlessness",
    "rust colored phlegm": "rusty_sputum",
    "rusty sputum": "rusty_sputum",
    "night sweats": "night_sweats",
    "chronic fatigue": "chronic_fatigue",
    "coughing at night": "coughing_at_night",
    "barking cough": "barking_cough",
    "bladder discomfort": "bladder_discomfort",
    "urinary urgency": "urinary_urgency",
    "foul smell of urine": "foul_smell_of_urine",
    "cloudy urine": "cloudy_urine",
    "yellowish skin": "yellowish_skin",
    "yellowing of eyes": "yellowing_of_eyes",
    "red spots over body": "red_spots_over_body",
    "silver like dusting": "silver_like_dusting",
    "small dents in nails": "small_dents_in_nails",
    "inflammatory nails": "inflammatory_nails",
    "joint redness": "joint_redness",
    "joint warmth": "joint_warmth",
    "foot pain": "foot_pain",
    "knee pain": "knee_pain",
    "hip joint pain": "hip_joint_pain",
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
    "cold hands and feet": "cold_hands_and_feets",
    "cold feet": "cold_hands_and_feets",
    "abdominal cramping": "abdominal_cramping",
    "abdominal bloating": "abdominal_bloating",
    "facial pain": "facial_pain",
    "facial pressure": "facial_pressure",
    "sore throat": "sore_throat",
    "painful swallowing": "painful_swallowing",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "lymph nodes": "swelled_lymph_nodes",
    "patches in throat": "patches_in_throat",
    "white patches on tongue": "white_patches_on_tongue",
    "white patches on throat": "white_patches_on_tongue",
    "skin tenderness": "skin_tenderness",
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
    "mood swings": "mood_swings",
    "puffy face and eyes": "puffy_face_and_eyes",
    "puffy face": "puffy_face_and_eyes",
    "hair loss": "hair_loss",
    "thinning hair": "hair_loss",
    "finger swelling": "finger_swelling",
    "back pain": "back_pain",
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
    "excessive hunger": "excessive_hunger",
    "blurred vision": "blurred_and_distorted_vision",
    "visual disturbances": "visual_disturbances",

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
    "anxiety": "anxiety",
    "restlessness": "restlessness",
    "lethargy": "lethargy",
    "cough": "cough",
    "coughing": "cough",
    "sweating": "sweating",
    "sweat": "sweating",
    "dehydration": "dehydration",
    "indigestion": "indigestion",
    "migraine": "headache",
    "nausea": "nausea",
    "nauseous": "nausea",
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
    phrase matching, conversational mappings, and lemmatized token lookup.
    """
    if not text or not text.strip():
        return []

    text_clean = text.lower().replace("'", "").replace("’", "")
    text_clean = re.sub(r"[^\w\s]", " ", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    matched = set()
    remaining_text = text_clean

    # Step 1: Match multi-word phrases & conversational mappings (longest first)
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)

    for kw in sorted_keywords:
        kw_clean = kw.replace("'", "")
        pattern = r"\b" + re.escape(kw_clean) + r"\b"
        if re.search(pattern, remaining_text):
            symptom = _KEYWORD_MAP[kw]
            if symptom in valid_symptoms:
                matched.add(symptom)
            remaining_text = re.sub(pattern, " ", remaining_text)

    # Step 2: Lemmatize / de-pluralize remaining tokens
    tokens = remaining_text.split()
    for token in tokens:
        if token in _STOP_WORDS or token in _GENERIC_WORDS or len(token) < 3:
            continue

        lemmas = [token]
        if token.endswith("s") and len(token) > 3:
            lemmas.append(token[:-1])
        if token.endswith("es") and len(token) > 4:
            lemmas.append(token[:-2])
        if token.endswith("ies") and len(token) > 4:
            lemmas.append(token[:-3] + "y")

        for lem in lemmas:
            if lem in valid_symptoms:
                matched.add(lem)
            elif lem in _KEYWORD_MAP:
                sym = _KEYWORD_MAP[lem]
                if sym in valid_symptoms:
                    matched.add(sym)

    return sorted(matched)
