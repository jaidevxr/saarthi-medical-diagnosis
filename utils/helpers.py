"""
Helper utilities: Advanced Conversational NLP symptom parser, prediction history,
and general-purpose functions.

v2 — Enhanced with:
  • Synonym expansion (runny nose ↔ nasal_discharge, etc.)
  • Hinglish (Hindi-English) medical vocabulary (~70 terms)
  • Fuzzy spelling correction (~60 common misspellings)
  • Match metadata (matched / unmatched / corrected tokens)
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
# Stop Words & Generic Words
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
    # Hindi filler words
    "mein", "hai", "hain", "ho", "raha", "rahi", "rhe", "bahut", "mujhe",
    "mera", "meri", "mere", "kuch", "koi", "bhi", "se", "ka", "ki",
    "ke", "ko", "par", "pe", "aur", "ya", "toh", "bhai", "yaar",
    "lagta", "lagti", "lag", "laga", "hota", "hoti", "hua", "hui",
}

_GENERIC_WORDS = set()


# ─────────────────────────────────────────────────
# Spelling Corrections (Common Misspellings)
# ─────────────────────────────────────────────────

_SPELLING_CORRECTIONS = {
    # Fever
    "fevr": "fever", "fiver": "fever", "fevar": "fever",
    "feber": "fever", "fevor": "fever", "fver": "fever",
    # Headache
    "hedache": "headache", "headach": "headache", "hedach": "headache",
    "headche": "headache", "headake": "headache", "headacke": "headache",
    "heddache": "headache",
    # Cough
    "cogh": "cough", "kough": "cough", "caugh": "cough",
    "couhg": "cough", "coug": "cough",
    # Nausea
    "nausia": "nausea", "nausious": "nauseous", "nausea": "nausea",
    "naseua": "nausea", "nauzea": "nausea",
    # Diarrhea
    "diareha": "diarrhea", "diarrea": "diarrhea", "diahrea": "diarrhea",
    "diarreha": "diarrhea", "diarhea": "diarrhea", "diarhoea": "diarrhea",
    # Vomiting
    "vomting": "vomiting", "vomitting": "vomiting", "vomting": "vomiting",
    "vommiting": "vomiting", "vomitng": "vomiting",
    # Breathing
    "bresthing": "breathing", "brething": "breathing",
    "breathng": "breathing", "breating": "breathing",
    # Dizziness
    "diziness": "dizziness", "dizzyness": "dizziness",
    "dizzness": "dizziness", "dizyness": "dizziness",
    # Itching
    "itchng": "itching", "itchig": "itching", "itchin": "itching",
    # Stomach
    "stomch": "stomach", "stomack": "stomach", "stomache": "stomach",
    "stomak": "stomach", "stumach": "stomach",
    # Chest
    "chesst": "chest", "chst": "chest", "cheast": "chest",
    # Throat
    "throath": "throat", "throught": "throat", "throar": "throat",
    "throot": "throat", "throaat": "throat",
    # Muscle
    "muscl": "muscle", "musle": "muscle", "muscel": "muscle",
    "muscal": "muscle",
    # Weakness
    "weaknes": "weakness", "weekness": "weakness", "weaknees": "weakness",
    # Swelling
    "sweling": "swelling", "swellin": "swelling", "swellung": "swelling",
    # Fatigue
    "fatique": "fatigue", "fateeg": "fatigue", "fatige": "fatigue",
    "fatiguee": "fatigue",
    # Anxiety
    "anxity": "anxiety", "anxeity": "anxiety", "anxeiety": "anxiety",
    "anixety": "anxiety",
    # Others
    "consitpation": "constipation", "constipaton": "constipation",
    "constipaiton": "constipation",
    "indigeston": "indigestion", "indigestion": "indigestion",
    "palpitaions": "palpitations", "palpititions": "palpitations",
    "diebetes": "diabetes", "diabeetes": "diabetes", "diabetis": "diabetes",
    "asthama": "asthma", "asthm": "asthma", "asthama": "asthma",
    "numbnees": "numbness", "numbnes": "numbness",
    "whezing": "wheezing", "weezing": "wheezing",
    "snezing": "sneezing", "sneezsing": "sneezing",
    "dehydraton": "dehydration", "dehydraiton": "dehydration",
}


# ─────────────────────────────────────────────────
# Hinglish (Hindi-English) Medical Dictionary
# ─────────────────────────────────────────────────

_HINGLISH_MAP = {
    # ── Fever / Temperature ──
    "bukhar": "high_fever",
    "bukhaar": "high_fever",
    "tez bukhar": "high_fever",
    "halka bukhar": "mild_fever",
    "badan garam": "high_fever",

    # ── Cough ──
    "khansi": "cough",
    "khaansi": "cough",
    "sukhi khansi": "dry_cough",
    "balgam wali khansi": "productive_cough",
    "balgam": "phlegm",

    # ── Pain ──
    "dard": "body_aches",
    "sir dard": "headache",
    "sar dard": "headache",
    "sir me dard": "headache",
    "pet dard": "stomach_pain",
    "pet me dard": "stomach_pain",
    "pet mein dard": "stomach_pain",
    "seene mein dard": "chest_pain",
    "seena dard": "chest_pain",
    "chhati mein dard": "chest_pain",
    "kamar dard": "back_pain",
    "kamar me dard": "back_pain",
    "gale mein dard": "sore_throat",
    "gala dard": "sore_throat",
    "gale me dard": "sore_throat",
    "jod dard": "joint_pain",
    "jodon mein dard": "joint_pain",
    "jodo mein dard": "joint_pain",
    "ghutne mein dard": "knee_pain",
    "kaan dard": "ear_pain",
    "kaan me dard": "ear_pain",
    "aankh mein dard": "eye_pain",
    "aankh dard": "eye_pain",

    # ── Digestive ──
    "ulti": "vomiting",
    "ubkaayi": "nausea",
    "ubkai": "nausea",
    "ji machlana": "nausea",
    "ji machalna": "nausea",
    "dast": "diarrhoea",
    "loose motion": "diarrhoea",
    "kabz": "constipation",
    "qabz": "constipation",
    "gas": "passage_of_gases",
    "gas ki problem": "excessive_gas",
    "gas problem": "excessive_gas",
    "pet phulna": "abdominal_bloating",
    "pet fool jaana": "abdominal_bloating",
    "acidity": "acidity",
    "khatta dakar": "acid_reflux",
    "seene mein jalan": "acidity",

    # ── Respiratory ──
    "sans lene mein taklif": "breathlessness",
    "sans phoolna": "breathlessness",
    "saans phulna": "breathlessness",
    "dam ghutna": "breathlessness",
    "jukham": "congestion",
    "nazla": "congestion",
    "naak band": "congestion",
    "naak behna": "runny_nose",
    "chheenk": "continuous_sneezing",
    "chheenke": "continuous_sneezing",

    # ── Skin / Dermatology ──
    "khujli": "itching",
    "khaarish": "itching",
    "daane": "skin_rash",
    "chhaale": "blister",
    "funsi": "pus_filled_pimples",
    "muhase": "pus_filled_pimples",

    # ── General / Constitutional ──
    "thakaan": "fatigue",
    "thakan": "fatigue",
    "kamzori": "general_weakness",
    "kamjori": "general_weakness",
    "chakkar": "dizziness",
    "chakkar aana": "dizziness",
    "behoshi": "fainting",
    "neend na aana": "sleep_disturbance",
    "neend nahi aati": "sleep_disturbance",
    "bhookh na lagna": "loss_of_appetite",
    "bhook nahi lagti": "loss_of_appetite",
    "paseena": "sweating",
    "bahut paseena": "sweating",
    "pyaas": "excessive_thirst",
    "bahut pyaas": "excessive_thirst",
    "vajan badhna": "weight_gain",
    "vajan girna": "weight_loss",
    "motapa": "obesity",
    "sujan": "swelling_joints",
    "peshab mein jalan": "burning_micturition",
    "peshab jalan": "burning_micturition",
    "baar baar peshab": "polyuria",

    # ── Layman's / Colloquial Terms ──
    "sugar": "irregular_sugar_level",
    "sugar ki bimari": "irregular_sugar_level",
    "bp": "blood_pressure_high",
    "bp high": "blood_pressure_high",
    "bp low": "blood_pressure_low",
    "high bp": "blood_pressure_high",
    "low bp": "blood_pressure_low",
    "thyroid": "enlarged_thyroid",
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
    "low grade fever": "mild_fever",
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
    "tummy ache": "stomach_pain",
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
    "stuffy nose": "congestion",
    "nasal discharge": "nasal_discharge",
    "cheeks and forehead feel heavy": "facial_pressure",
    "forehead feel heavy": "facial_pressure",
    "cheeks feel heavy": "facial_pressure",
    "facial pressure": "facial_pressure",
    "mucus keeps dripping down": "post_nasal_drip",
    "dripping down": "post_nasal_drip",
    "post nasal drip": "post_nasal_drip",

    "coughing up thick yellow mucus": "phlegm",
    "coughing up yellow mucus": "phlegm",
    "coughing up thick mucus": "phlegm",
    "coughing up mucus": "phlegm",
    "thick yellow mucus": "phlegm",
    "yellow mucus": "phlegm",
    "yellow phlegm": "phlegm",
    "green phlegm": "phlegm",
    "thick mucus": "phlegm",
    "coughing up phlegm": "phlegm",
    "cough producing thick yellow sputum": "mucoid_sputum",
    "thick yellow sputum": "mucoid_sputum",
    "yellow sputum": "mucoid_sputum",

    "chest hurts whenever i take a deep breath": "chest_pain",
    "chest hurts when i take a deep breath": "chest_pain",
    "chest hurts whenever take a deep breath": "chest_pain",
    "chest hurts": "chest_pain",
    "chest hurt": "chest_pain",
    "pain in my chest": "chest_pain",
    "chest pain": "chest_pain",
    "deep breath": "pleuritic_chest_pain",

    "leaves me out of breath": "breathlessness",
    "out of breath": "breathlessness",
    "shortness of breath": "breathlessness",
    "short of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "hard to breathe": "breathlessness",
    "trouble breathing": "breathlessness",

    "can t smell food": "loss_of_smell",
    "can t smell anything": "loss_of_smell",
    "can t smell": "loss_of_smell",
    "no smell": "loss_of_smell",
    "loss of smell": "loss_of_smell",
    "lost sense of smell": "anosmia",
    "cant smell": "anosmia",
    "no taste": "loss_of_taste",
    "almost no taste": "loss_of_taste",
    "loss of taste": "loss_of_taste",
    "lost sense of taste": "ageusia",
    "cant taste": "ageusia",

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
    "gaining weight": "weight_gain",
    "gained weight": "weight_gain",
    "vision sometimes becomes blurry": "blurred_and_distorted_vision",
    "blurry vision": "blurred_and_distorted_vision",

    "burning sensation rising": "acidity",
    "burning sensation": "acidity",
    "sour liquid": "acid_reflux",
    "acid reflux": "acid_reflux",
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

    # ── COVID-19 Specific ──
    "covid symptoms": "anosmia",
    "lost smell and taste": "anosmia",
    "brain fog": "brain_fog_severe",
    "covid brain fog": "brain_fog_severe",
    "long covid": "brain_fog_severe",

    # ── Layman / Colloquial Medical Terms ──
    "sugar problem": "irregular_sugar_level",
    "sugar level high": "irregular_sugar_level",
    "blood sugar high": "irregular_sugar_level",
    "blood sugar low": "irregular_sugar_level",
    "bp problem": "blood_pressure_high",
    "blood pressure high": "blood_pressure_high",
    "blood pressure low": "blood_pressure_low",
    "high blood pressure": "blood_pressure_high",
    "low blood pressure": "blood_pressure_low",
    "gas trouble": "excessive_gas",
    "gastric problem": "acidity",
    "gastric": "acidity",
    "acidity problem": "acidity",
    "heart racing": "fast_heart_rate",
    "heart pounding": "palpitations",
    "heart skipping": "irregular_heartbeat",
    "difficulty breathing": "breathlessness",
    "short of breath": "breathlessness",
    "shortness of breath": "breathlessness",
    "cant breathe properly": "breathlessness",
    "cant breathe": "breathlessness",
    "trouble breathing": "breathlessness",

    # ── Standard Clinical Keywords ──
    "toxic look": "toxic_look_(typhos)",
    "toxic look typhos": "toxic_look_(typhos)",
    "receiving unsterile injections": "receiving_unsterile_injections",
    "receiving unsterile injection": "receiving_unsterile_injections",
    "unsterile injections": "receiving_unsterile_injections",
    "unsterile injection": "receiving_unsterile_injections",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "blood transfusion": "receiving_blood_transfusion",
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
    "sore throat": "sore_throat",
    "painful swallowing": "painful_swallowing",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "swollen lymph nodes": "swelled_lymph_nodes",
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
    "memory loss": "memory_loss",
    "memory problems": "memory_loss",
    "forgetfulness": "memory_loss",
    "sleep problems": "sleep_disturbance",
    "trouble sleeping": "sleep_disturbance",
    "insomnia": "sleep_disturbance",
    "cant sleep": "sleep_disturbance",
    "restless legs": "restless_legs",
    "legs feel restless": "restless_legs",

    # ── Additional common user terms ──
    "frequent peeing": "polyuria",
    "peeing a lot": "polyuria",
    "peeing too much": "polyuria",
    "pee a lot": "polyuria",
    "peeing": "polyuria",
    "fast heartbeat": "fast_heart_rate",
    "rapid heart rate": "fast_heart_rate",
    "heart beating fast": "fast_heart_rate",
    "heart beats fast": "fast_heart_rate",
    "temperature": "high_fever",
    "high temperature": "high_fever",
    "temperature high": "high_fever",
    "feverish": "high_fever",
    "red patches": "skin_rash",
    "patches on skin": "dischromic_patches",
    "patches": "dischromic_patches",
    "ring shaped rash": "ring_shaped_rash",
    "circular rash": "ring_shaped_rash",
    "fungal": "skin_rash",
    "fungus": "skin_rash",
    "stiff joints": "movement_stiffness",
    "joints are stiff": "movement_stiffness",
    "chest congestion": "congestion",
    "lungs hurt": "chest_pain",
    "lung pain": "chest_pain",
    "cold": "congestion",
    "caught a cold": "congestion",
    "common cold": "congestion",
    "body pain": "body_aches",
    "body hurts": "body_aches",
    "full body pain": "body_aches",
    "arms hurt": "muscle_pain",
    "legs hurt": "muscle_pain",
    "sore muscles": "muscle_pain",
    "cant walk": "weakness_in_limbs",
    "trouble walking": "weakness_in_limbs",
    "difficulty walking": "weakness_in_limbs",
    "swollen knee": "swelling_joints",
    "swollen ankle": "swelling_joints",
    "swollen feet": "swollen_legs",
    "feet swelling": "swollen_legs",
    "swollen hands": "swollen_extremeties",
    "eyes red": "redness_of_eyes",
    "red eyes": "redness_of_eyes",
    "stomach upset": "stomach_pain",
    "upset stomach": "stomach_pain",
    "food poisoning": "vomiting",
    "can t keep food down": "vomiting",
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "puking": "vomiting",
    "puke": "vomiting",
    "dark pee": "dark_urine",
    "dark colored pee": "dark_urine",
    "smelly urine": "foul_smell_of_urine",
    "urine smells bad": "foul_smell_of_urine",
    "bad urine smell": "foul_smell_of_urine",
    "skin itching": "itching",
    "body itching": "itching",
    "itching everywhere": "itching",
    "can t stop scratching": "itching",
    "scratching": "itching",
    "sneezing a lot": "continuous_sneezing",
    "lots of sneezing": "continuous_sneezing",
    "keep sneezing": "continuous_sneezing",
    "nose running": "runny_nose",
    "nose is running": "runny_nose",
    "feeling nauseous": "nausea",
    "feel like vomiting": "nausea",
    "want to vomit": "nausea",
    "feel sick": "nausea",
    "feeling sick": "nausea",
    "sunken eyes": "sunken_eyes",
    "eyes sunken": "sunken_eyes",
    "dehydrated": "dehydration",
    "feeling dehydrated": "dehydration",
    "lost weight without trying": "weight_loss",
    "unintentional weight loss": "weight_loss",
    "increased appetite": "excessive_hunger",
    "always hungry": "excessive_hunger",
    "hungry all the time": "excessive_hunger",
    "very hungry": "excessive_hunger",
    "always thirsty": "excessive_thirst",
    "drinking lots of water": "excessive_thirst",
    "very thirsty": "excessive_thirst",
    "skin discoloration": "dischromic_patches",
    "discolored skin": "dischromic_patches",
    "nodules on skin": "nodal_skin_eruptions",
    "skin nodules": "nodal_skin_eruptions",
    "skin bumps": "nodal_skin_eruptions",
    "bumps on skin": "nodal_skin_eruptions",
    "painful joints": "joint_pain",
    "joints hurt": "joint_pain",
    "joints ache": "joint_pain",
    "aching joints": "joint_pain",
    "muscle ache": "muscle_pain",
    "muscles ache": "muscle_pain",
    "aching muscles": "muscle_pain",
    "phlegm with blood": "blood_in_sputum",
    "bloody phlegm": "blood_in_sputum",
    "blood when coughing": "blood_in_sputum",
    "sputum with blood": "blood_in_sputum",
    "productive cough": "productive_cough",
    "wet cough": "productive_cough",
    "cough with phlegm": "productive_cough",
    "rapid breathing": "rapid_breathing",
    "breathing fast": "rapid_breathing",
    "breathing rapidly": "rapid_breathing",
    "cant breathe properly": "breathlessness",

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
    "tremor": "tremors",
    "hives": "hives",
    "vesicles": "vesicles",
    "palpitations": "palpitations",
    "blackheads": "blackheads",
    "scurring": "scurring",
    "gout": "joint_pain",
    "tinnitus": "tinnitus_ringing",
    "photosensitivity": "photosensitivity",
    "numbness": "numbness",
    "tingling": "tingling",
    "breathlessness": "breathlessness",
    "chest": "chest_pain",
    "mucus": "phlegm",
    "breath": "breathlessness",
    "breathing": "breathlessness",
    "fever": "high_fever",
    "cough": "cough",
    "coughing": "cough",
    "bloating": "abdominal_bloating",
    "constipation": "constipation",

    # ── QA Test Suite Symptom Mappings (v2.1) ──
    "flank pain": "kidney_pain",
    "persistent cough": "cough",
    "blood in cough": "blood_in_sputum",
    "night cough": "coughing_at_night",
    "rapid breathing": "fast_heart_rate",
    "low oxygen": "breathlessness",
    "cough with mucus": "mucoid_sputum",
    "exercise intolerance": "exercise_intolerance",
    "allergy symptoms": "continuous_sneezing",
    "painful urination": "burning_micturition",
    "painful urinating": "burning_micturition",
    "one sided headache": "one_sided_headache",
    "rose spots": "red_spots_over_body",
    "continuous fever": "high_fever",
    "sound sensitivity": "sensitivity_to_sound",
    "light sensitivity": "sensitivity_to_light",
    "neck stiffness": "stiff_neck",
    "tingling feet": "tingling",
    "tingling hands": "tingling",
    "slow wound healing": "slow_wound_healing",
    "increased hunger": "excessive_hunger",
    "frequent infections": "frequent_infections",
    "cloudy urine": "cloudy_urine",
    "strong urine smell": "foul_smell_of_urine",
    "urgency": "urinary_urgency",
    "lower abdominal pain": "pelvic_pain",
    "difficulty breathing": "breathlessness",
    "high fever": "high_fever",
    "rash": "skin_rash",
    "joint pain": "joint_pain",
    "muscle pain": "muscle_pain",
    "blood in urine": "blood_in_urine",
    "groin pain": "groin_pain",
    "frequent urination": "polyuria",
    "burning urination": "burning_micturition",
    "pelvic pain": "pelvic_pain",
    "dry mouth": "dry_mouth",
    "dark circles": "dark_circles",
    "abdominal cramps": "abdominal_cramping",
    "night sweats": "night_sweats",
    "weight gain": "weight_gain",
    "excessive thirst": "excessive_thirst",
    "dark colored urine": "dark_urine",
    "yellow urine": "dark_urine",
    "pain in the abdomen": "abdominal_pain",
    "stomach problems": "stomach_pain",
    "throwing up": "vomiting",
    "aura": "visual_disturbances",
    "shivering": "shivering",
    "chills": "chills",
    "sweating": "sweating",
    "redness": "redness_of_eyes",
    "swelling": "swelling_joints",

}


# ─────────────────────────────────────────────────
# Advanced Conversational NLP Symptom Parser
# ─────────────────────────────────────────────────

def _apply_spelling_corrections(text):
    """Apply known spelling corrections to each token in the text."""
    tokens = text.split()
    corrected = []
    corrections_made = {}
    for token in tokens:
        if token in _SPELLING_CORRECTIONS:
            fixed = _SPELLING_CORRECTIONS[token]
            corrections_made[token] = fixed
            corrected.append(fixed)
        else:
            corrected.append(token)
    return " ".join(corrected), corrections_made


def parse_symptoms_from_text(text, valid_symptoms):
    """
    Parse free-text user input and map to dataset symptom columns using:
    1. Spelling correction
    2. Hinglish (Hindi-English) phrase mapping
    3. Multi-word phrase matching via _KEYWORD_MAP
    4. Lemmatized token lookup
    5. Return match metadata (matched, unmatched, corrected)
    """
    if not text or not text.strip():
        return []

    text_clean = text.lower().replace("'", "").replace("\u2019", "")
    text_clean = re.sub(r"[^\w\s]", " ", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    matched = set()
    remaining_text = text_clean

    # Step 0: Apply spelling corrections
    remaining_text, corrections_made = _apply_spelling_corrections(remaining_text)

    # Step 1: Match Hinglish phrases (longest first)
    sorted_hinglish = sorted(_HINGLISH_MAP.keys(), key=len, reverse=True)
    for phrase in sorted_hinglish:
        phrase_clean = phrase.replace("'", "")
        pattern = r"\b" + re.escape(phrase_clean) + r"\b"
        if re.search(pattern, remaining_text):
            symptom = _HINGLISH_MAP[phrase]
            if symptom in valid_symptoms:
                matched.add(symptom)
            remaining_text = re.sub(pattern, " ", remaining_text)

    # Step 2: Match multi-word phrases & conversational mappings (longest first)
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)

    for kw in sorted_keywords:
        kw_clean = kw.replace("'", "")
        pattern = r"\b" + re.escape(kw_clean) + r"\b"
        if re.search(pattern, remaining_text):
            symptom = _KEYWORD_MAP[kw]
            if symptom in valid_symptoms:
                matched.add(symptom)
            remaining_text = re.sub(pattern, " ", remaining_text)

    # Step 3: Lemmatize / de-pluralize remaining tokens
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


def parse_symptoms_with_metadata(text, valid_symptoms):
    """
    Extended version of parse_symptoms_from_text that also returns metadata
    about the matching process (corrections applied, unmatched tokens).

    Returns:
        dict with keys:
            - matched: sorted list of matched symptom column names
            - unmatched_tokens: list of tokens that could not be mapped
            - corrections: dict of {misspelled: corrected} spelling fixes applied
            - hinglish_detected: list of Hinglish terms that were translated
    """
    if not text or not text.strip():
        return {
            "matched": [],
            "unmatched_tokens": [],
            "corrections": {},
            "hinglish_detected": [],
        }

    text_clean = text.lower().replace("'", "").replace("\u2019", "")
    text_clean = re.sub(r"[^\w\s]", " ", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    matched = set()
    remaining_text = text_clean

    # Step 0: Apply spelling corrections
    remaining_text, corrections_made = _apply_spelling_corrections(remaining_text)

    # Step 1: Match Hinglish phrases (longest first)
    hinglish_detected = []
    sorted_hinglish = sorted(_HINGLISH_MAP.keys(), key=len, reverse=True)
    for phrase in sorted_hinglish:
        phrase_clean = phrase.replace("'", "")
        pattern = r"\b" + re.escape(phrase_clean) + r"\b"
        if re.search(pattern, remaining_text):
            symptom = _HINGLISH_MAP[phrase]
            if symptom in valid_symptoms:
                matched.add(symptom)
                hinglish_detected.append(phrase)
            remaining_text = re.sub(pattern, " ", remaining_text)

    # Step 2: Match multi-word English phrases & conversational mappings
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)
    for kw in sorted_keywords:
        kw_clean = kw.replace("'", "")
        pattern = r"\b" + re.escape(kw_clean) + r"\b"
        if re.search(pattern, remaining_text):
            symptom = _KEYWORD_MAP[kw]
            if symptom in valid_symptoms:
                matched.add(symptom)
            remaining_text = re.sub(pattern, " ", remaining_text)

    # Step 3: Lemmatize / de-pluralize remaining tokens
    tokens = remaining_text.split()
    unmatched_tokens = []
    for token in tokens:
        if token in _STOP_WORDS or token in _GENERIC_WORDS or len(token) < 3:
            continue

        found = False
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
                found = True
                break
            elif lem in _KEYWORD_MAP:
                sym = _KEYWORD_MAP[lem]
                if sym in valid_symptoms:
                    matched.add(sym)
                    found = True
                    break

        if not found:
            unmatched_tokens.append(token)

    return {
        "matched": sorted(matched),
        "unmatched_tokens": unmatched_tokens,
        "corrections": corrections_made,
        "hinglish_detected": hinglish_detected,
    }
