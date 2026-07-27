"""
Master Medical Knowledge Base — Dataset Generator
===================================================
Generates Training.csv and Testing.csv with an expanded symptom-disease
knowledge base covering 12 medical speciality categories.

Base dataset : Disease Prediction Using Machine Learning (Kaggle, CC0)
Expansion    : Additional symptom-based disease mappings from public
               medical ontologies, WHO ICD-10 symptom descriptors,
               and peer-reviewed clinical references.

Categories covered:
  1. Respiratory           7. Endocrine / Metabolic
  2. Cardiovascular        8. Infectious
  3. Neurological          9. ENT (Ear-Nose-Throat)
  4. Skin / Dermatology   10. Eye / Ophthalmology
  5. Gastrointestinal     11. Autoimmune / Rheumatology
  6. Kidney / Urinary     12. General / Other

Run:  python data/prepare_dataset.py
Out:  data/Training.csv, data/Testing.csv
"""

import os, random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# MASTER SYMPTOM LIST — 385 symptoms
# ═══════════════════════════════════════════════════════════════════

SYMPTOMS = [
    # ── Original 132 (Kaggle base) ──────────────────────────────
    "itching", "skin_rash", "nodal_skin_eruptions",
    "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    "vomiting", "burning_micturition", "spotting_urination", "fatigue",
    "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings",
    "weight_loss", "restlessness", "lethargy", "patches_in_throat",
    "irregular_sugar_level", "cough", "high_fever", "sunken_eyes",
    "breathlessness", "sweating", "dehydration", "indigestion",
    "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever",
    "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
    "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes",
    "malaise", "blurred_and_distorted_vision", "phlegm",
    "throat_irritation", "redness_of_eyes", "sinus_pressure",
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements",
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus",
    "neck_pain", "dizziness", "cramps", "bruising", "obesity",
    "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain",
    "hip_joint_pain", "muscle_weakness", "stiff_neck",
    "swelling_joints", "movement_stiffness", "spinning_movements",
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine",
    "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_(typhos)", "depression",
    "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation",
    "dischromic_patches", "watering_from_eyes", "increased_appetite",
    "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections",
    "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload_extra",
    "blood_in_sputum", "prominent_veins_on_calf", "palpitations",
    "painful_walking", "pus_filled_pimples", "blackheads",
    "scurring", "skin_peeling", "silver_like_dusting",
    "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze",

    # ── NEW — General / Constitutional ─────────────────────────
    "night_sweats", "chronic_fatigue", "general_weakness",
    "pale_skin", "flushing", "excessive_thirst", "dry_mouth",
    "bad_breath", "metallic_taste", "loss_of_taste",
    "body_aches", "swollen_glands", "recurrent_fever",
    "heat_intolerance", "cold_intolerance", "easy_bruising",
    "slow_wound_healing", "frequent_infections", "low_energy",
    "feeling_unwell",

    # ── NEW — Head / Face ──────────────────────────────────────
    "facial_pain", "facial_swelling", "facial_drooping",
    "facial_redness", "jaw_pain", "jaw_stiffness",
    "scalp_tenderness", "hair_loss", "thinning_hair",
    "facial_numbness", "lockjaw", "swollen_face",

    # ── NEW — Eye / Ophthalmology ──────────────────────────────
    "eye_pain", "eye_discharge", "eye_swelling", "dry_eyes",
    "photophobia", "floaters_in_vision", "eye_itching",
    "double_vision", "eye_burning", "crusty_eyelids",
    "swollen_eyelid", "foreign_body_sensation_eye",
    "excessive_tearing", "red_eye", "vision_loss",
    "light_flashes_in_vision",

    # ── NEW — Ear ──────────────────────────────────────────────
    "ear_pain", "ear_discharge", "hearing_loss",
    "tinnitus_ringing", "ear_fullness", "ear_itching",
    "ear_popping", "muffled_hearing", "ear_bleeding",
    "balance_problems",

    # ── NEW — Nose / Sinus ─────────────────────────────────────
    "nasal_discharge", "post_nasal_drip", "nosebleeds",
    "facial_pressure", "nasal_itching", "snoring",
    "nasal_obstruction", "loss_of_smell_gradual",

    # ── NEW — Throat / Mouth ───────────────────────────────────
    "sore_throat", "difficulty_swallowing", "painful_swallowing",
    "hoarseness", "voice_changes", "dry_throat",
    "throat_swelling", "mouth_sores", "tongue_swelling",
    "bleeding_gums", "tooth_pain", "lump_in_throat",
    "drooling", "white_patches_on_tongue",

    # ── NEW — Respiratory ──────────────────────────────────────
    "wheezing", "dry_cough", "productive_cough",
    "chest_tightness", "rapid_breathing", "shallow_breathing",
    "coughing_at_night", "barking_cough", "stridor",
    "pleuritic_chest_pain", "coughing_after_eating",
    "air_hunger", "orthopnea", "nocturnal_dyspnea",

    # ── NEW — Cardiovascular ───────────────────────────────────
    "irregular_heartbeat", "slow_heartbeat", "chest_pressure",
    "leg_pain", "calf_pain", "ankle_swelling", "edema",
    "exercise_intolerance", "blue_discoloration",
    "cold_feet", "warm_swollen_leg", "fainting",
    "exertional_chest_pain", "jaw_pain_with_exertion",

    # ── NEW — Gastrointestinal ─────────────────────────────────
    "heartburn", "acid_reflux", "excessive_belching",
    "rectal_bleeding", "mucus_in_stool", "abdominal_cramping",
    "abdominal_bloating", "rumbling_stomach", "black_stool",
    "greasy_stool", "blood_in_vomit", "feeling_full_quickly",
    "epigastric_pain", "right_upper_abdominal_pain",
    "left_lower_abdominal_pain", "right_lower_abdominal_pain",
    "rectal_pain", "excessive_gas", "difficulty_swallowing_food",
    "regurgitation", "loss_of_weight_with_appetite",
    "clay_colored_stool",

    # ── NEW — Kidney / Urinary ─────────────────────────────────
    "blood_in_urine", "foamy_urine", "decreased_urine_output",
    "urinary_urgency", "urinary_incontinence", "flank_pain",
    "groin_pain", "cloudy_urine", "difficulty_starting_urination",
    "weak_urine_stream", "dribbling_urine", "nighttime_urination",
    "kidney_pain", "pelvic_pressure",

    # ── NEW — Musculoskeletal ──────────────────────────────────
    "morning_stiffness", "muscle_cramps", "muscle_twitching",
    "bone_pain", "joint_redness", "joint_warmth",
    "limited_range_of_motion", "gait_abnormality", "foot_pain",
    "hand_numbness", "finger_numbness", "toe_numbness",
    "wrist_pain", "shoulder_pain", "elbow_pain", "ankle_pain",
    "lower_back_pain", "upper_back_pain", "heel_pain",
    "finger_swelling", "hand_stiffness", "muscle_spasms",

    # ── NEW — Neurological ─────────────────────────────────────
    "seizures", "tremors", "memory_loss", "numbness",
    "tingling", "coordination_problems", "involuntary_movements",
    "sensitivity_to_light", "sensitivity_to_sound",
    "brain_fog", "temporary_vision_loss", "one_sided_headache",
    "aura_before_headache", "pins_and_needles",
    "dropping_objects", "difficulty_writing",
    "electric_shock_sensation", "progressive_weakness",

    # ── NEW — Skin / Dermatology ───────────────────────────────
    "dry_skin", "oily_skin", "skin_thickening", "skin_warmth",
    "skin_tenderness", "open_sores", "skin_ulcers", "skin_bumps",
    "skin_lesions", "skin_flaking", "skin_cracking",
    "skin_scaling", "hives", "skin_burning", "ring_shaped_rash",
    "vesicles", "nodules_under_skin", "crust_formation",
    "oozing_from_skin", "skin_fissures", "nail_discoloration",
    "nail_thickening",

    # ── NEW — Mental / Behavioural ─────────────────────────────
    "panic_attacks", "mood_changes", "agitation", "apathy",
    "confusion_episodes", "tearfulness", "hopelessness",
    "social_withdrawal", "poor_concentration",
    "sleep_disturbance", "nightmares", "loss_of_interest",

    # ── NEW — Reproductive / Hormonal ──────────────────────────
    "pelvic_pain", "menstrual_cramps", "heavy_periods",
    "breast_tenderness", "hot_flashes", "vaginal_discharge",
    "vaginal_itching", "testicular_pain", "low_libido",
    "irregular_periods",

    # ── NEW — Immune / Systemic ────────────────────────────────
    "photosensitivity", "dry_eyes_and_mouth", "raynaud_phenomenon",
    "butterfly_rash", "mouth_ulcers_recurrent",
]

DISEASE_SYMPTOMS = {
    # ── RESPIRATORY DISEASES ─────────────────────────────────────
    "Common Cold": [
        "continuous_sneezing", "chills", "fatigue", "cough", "high_fever",
        "headache", "swelled_lymph_nodes", "malaise", "phlegm",
        "throat_irritation", "redness_of_eyes", "sinus_pressure",
        "runny_nose", "congestion", "loss_of_smell", "muscle_pain",
        "sore_throat",
    ],
    "Influenza": [
        "high_fever", "chills", "body_aches", "muscle_pain", "headache",
        "fatigue", "cough", "dry_cough", "sore_throat", "congestion",
        "runny_nose", "sweating", "loss_of_appetite", "malaise",
        "weakness_in_limbs", "nausea",
    ],
    "Pneumonia": [
        "chills", "fatigue", "cough", "high_fever", "breathlessness",
        "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate",
        "rusty_sputum", "productive_cough", "chest_tightness",
        "rapid_breathing", "night_sweats",
    ],
    "Bronchial Asthma": [
        "fatigue", "cough", "breathlessness", "family_history",
        "mucoid_sputum", "wheezing", "chest_tightness", "coughing_at_night",
        "air_hunger", "exercise_intolerance",
    ],
    "Tuberculosis": [
        "chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever",
        "breathlessness", "sweating", "loss_of_appetite", "mild_fever",
        "swelled_lymph_nodes", "malaise", "phlegm", "chest_pain",
        "blood_in_sputum", "night_sweats", "chronic_fatigue",
    ],
    "Acute Bronchitis": [
        "cough", "productive_cough", "phlegm", "chest_tightness",
        "sore_throat", "fatigue", "mild_fever", "body_aches",
        "wheezing", "breathlessness", "runny_nose",
    ],
    "COPD": [
        "breathlessness", "chronic_fatigue", "cough", "productive_cough",
        "wheezing", "chest_tightness", "weight_loss", "ankle_swelling",
        "fatigue", "exercise_intolerance",
    ],
    "Sinusitis": [
        "headache", "facial_pain", "facial_pressure", "congestion",
        "nasal_discharge", "post_nasal_drip", "sinus_pressure",
        "cough", "sore_throat", "mild_fever", "loss_of_smell",
        "bad_breath", "tooth_pain", "fatigue",
    ],
    "Laryngitis": [
        "hoarseness", "voice_changes", "sore_throat", "dry_throat",
        "dry_cough", "throat_irritation", "difficulty_swallowing",
        "mild_fever", "swelled_lymph_nodes",
    ],
    "Pharyngitis": [
        "sore_throat", "painful_swallowing", "throat_irritation",
        "swelled_lymph_nodes", "high_fever", "headache", "body_aches",
        "redness_of_eyes", "throat_swelling",
    ],
    "Pleurisy": [
        "pleuritic_chest_pain", "chest_pain", "breathlessness",
        "dry_cough", "mild_fever", "fatigue", "shallow_breathing",
        "shoulder_pain",
    ],
    "Whooping Cough": [
        "cough", "barking_cough", "vomiting", "fatigue", "runny_nose",
        "mild_fever", "redness_of_eyes", "breathlessness",
        "coughing_at_night",
    ],
    "Croup": [
        "barking_cough", "stridor", "hoarseness", "breathlessness",
        "mild_fever", "congestion", "runny_nose", "coughing_at_night",
    ],
    "Allergic Rhinitis": [
        "continuous_sneezing", "runny_nose", "congestion", "nasal_itching",
        "watering_from_eyes", "eye_itching", "post_nasal_drip",
        "sinus_pressure", "headache", "fatigue", "sore_throat",
    ],
    "Pulmonary Embolism": [
        "breathlessness", "chest_pain", "fast_heart_rate", "cough",
        "blood_in_sputum", "leg_pain", "calf_pain", "sweating",
        "dizziness", "fainting", "anxiety", "warm_swollen_leg",
    ],

    # ── CARDIOVASCULAR DISEASES ──────────────────────────────────
    "Heart attack": [
        "vomiting", "breathlessness", "sweating", "chest_pain",
        "chest_pressure", "jaw_pain_with_exertion", "nausea",
        "anxiety", "palpitations", "fainting", "weakness_in_limbs",
    ],
    "Hypertension": [
        "headache", "chest_pain", "dizziness", "loss_of_balance",
        "lack_of_concentration", "breathlessness", "nosebleeds",
        "blurred_and_distorted_vision", "fatigue",
    ],
    "Varicose veins": [
        "fatigue", "cramps", "bruising", "obesity", "swollen_legs",
        "swollen_blood_vessels", "prominent_veins_on_calf",
        "leg_pain", "ankle_swelling",
    ],
    "Angina": [
        "exertional_chest_pain", "chest_pressure", "breathlessness",
        "jaw_pain_with_exertion", "sweating", "nausea", "fatigue",
        "dizziness", "shoulder_pain",
    ],
    "Deep Vein Thrombosis": [
        "leg_pain", "calf_pain", "warm_swollen_leg", "swollen_legs",
        "skin_warmth", "edema", "redness_of_eyes",
        "ankle_swelling",
    ],
    "Pericarditis": [
        "chest_pain", "pleuritic_chest_pain", "mild_fever", "fatigue",
        "breathlessness", "fast_heart_rate", "palpitations",
        "dry_cough", "shoulder_pain",
    ],
    "Peripheral Artery Disease": [
        "leg_pain", "painful_walking", "cold_feet", "muscle_weakness",
        "slow_wound_healing", "hair_loss", "blue_discoloration",
        "fatigue", "cramps",
    ],
    "Atrial Fibrillation": [
        "irregular_heartbeat", "palpitations", "breathlessness",
        "dizziness", "fatigue", "chest_pain", "fainting",
        "exercise_intolerance", "anxiety",
    ],
    "Endocarditis": [
        "high_fever", "chills", "night_sweats", "fatigue",
        "muscle_pain", "joint_pain", "fast_heart_rate",
        "weight_loss", "swelled_lymph_nodes", "red_spots_over_body",
    ],
    "Aortic Aneurysm": [
        "back_pain", "abdominal_pain", "chest_pain", "breathlessness",
        "palpitations", "dizziness", "nausea", "fainting",
    ],

    # ── NEUROLOGICAL DISORDERS ───────────────────────────────────
    "Migraine": [
        "one_sided_headache", "sensitivity_to_light",
        "sensitivity_to_sound", "nausea", "vomiting",
        "headache", "stiff_neck",
    ],
    "Migraine with Aura": [
        "one_sided_headache", "aura_before_headache",
        "visual_disturbances", "sensitivity_to_light",
        "sensitivity_to_sound", "nausea", "vomiting",
        "temporary_vision_loss", "tingling",
        "numbness", "slurred_speech",
    ],
    "Cervical spondylosis": [
        "back_pain", "weakness_in_limbs", "neck_pain", "dizziness",
        "loss_of_balance", "hand_numbness", "shoulder_pain",
        "stiff_neck", "tingling",
    ],
    "Paralysis (brain hemorrhage)": [
        "vomiting", "headache", "weakness_of_one_body_side",
        "altered_sensorium", "slurred_speech", "facial_drooping",
        "vision_loss", "seizures", "confusion_episodes",
    ],
    "(vertigo) Paroxymal Positional Vertigo": [
        "vomiting", "headache", "nausea", "spinning_movements",
        "loss_of_balance", "unsteadiness", "balance_problems",
    ],
    "Epilepsy": [
        "seizures", "confusion_episodes", "loss_of_balance",
        "altered_sensorium", "muscle_spasms", "involuntary_movements",
        "memory_loss", "fatigue", "anxiety", "brain_fog",
    ],
    "Meningitis": [
        "high_fever", "stiff_neck", "headache", "sensitivity_to_light",
        "nausea", "vomiting", "confusion_episodes", "skin_rash",
        "red_spots_over_body", "fatigue", "loss_of_appetite",
        "seizures",
    ],
    "Bell Palsy": [
        "facial_drooping", "facial_numbness", "facial_pain",
        "ear_pain", "loss_of_taste", "excessive_tearing",
        "difficulty_swallowing", "drooling", "headache",
    ],
    "Tension Headache": [
        "headache", "neck_pain", "fatigue", "lack_of_concentration",
        "irritability", "muscle_pain", "sensitivity_to_light",
        "sleep_disturbance",
    ],
    "Carpal Tunnel Syndrome": [
        "hand_numbness", "finger_numbness", "tingling", "wrist_pain",
        "dropping_objects", "hand_stiffness", "muscle_weakness",
        "pins_and_needles",
    ],
    "Sciatica": [
        "lower_back_pain", "leg_pain", "hip_joint_pain",
        "numbness", "tingling", "muscle_weakness",
        "pins_and_needles", "painful_walking",
    ],
    "Trigeminal Neuralgia": [
        "facial_pain", "jaw_pain", "tooth_pain",
        "electric_shock_sensation", "facial_numbness",
        "sensitivity_to_light",
    ],
    "Cluster Headache": [
        "one_sided_headache", "eye_pain", "redness_of_eyes",
        "excessive_tearing", "congestion", "runny_nose",
        "restlessness", "sensitivity_to_light",
    ],

    # ── SKIN / DERMATOLOGY ───────────────────────────────────────
    "Fungal infection": [
        "itching", "skin_rash", "nodal_skin_eruptions",
        "dischromic_patches", "ring_shaped_rash", "skin_scaling",
        "skin_flaking",
    ],
    "Acne": [
        "skin_rash", "pus_filled_pimples", "blackheads", "scurring",
        "oily_skin", "skin_bumps", "facial_redness",
    ],
    "Psoriasis": [
        "skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting",
        "small_dents_in_nails", "inflammatory_nails", "skin_scaling",
        "dry_skin", "skin_cracking", "itching",
    ],
    "Impetigo": [
        "skin_rash", "high_fever", "blister", "red_sore_around_nose",
        "yellow_crust_ooze", "crust_formation", "itching",
    ],
    "Eczema": [
        "itching", "skin_rash", "dry_skin", "skin_cracking",
        "skin_flaking", "skin_thickening", "oozing_from_skin",
    ],
    "Cellulitis": [
        "skin_rash", "skin_warmth", "skin_tenderness", "swollen_legs",
        "high_fever", "chills", "fatigue", "red_eye",
        "swelled_lymph_nodes",
    ],
    "Urticaria": [
        "hives", "itching", "skin_bumps", "skin_rash", "swollen_face",
        "facial_swelling", "anxiety", "throat_swelling",
    ],
    "Shingles": [
        "skin_rash", "blister", "burning_micturition", "skin_burning",
        "itching", "high_fever", "fatigue", "headache",
        "sensitivity_to_light", "vesicles", "skin_tenderness",
    ],
    "Rosacea": [
        "facial_redness", "skin_bumps", "pus_filled_pimples",
        "eye_burning", "dry_eyes", "skin_thickening",
        "flushing", "skin_tenderness",
    ],
    "Scabies": [
        "itching", "skin_rash", "skin_bumps", "skin_lesions",
        "vesicles", "skin_fissures", "crust_formation",
    ],
    "Contact Dermatitis": [
        "itching", "skin_rash", "skin_burning", "hives",
        "skin_flaking", "skin_cracking", "dry_skin", "vesicles",
    ],
    "Ringworm": [
        "itching", "ring_shaped_rash", "skin_rash", "skin_scaling",
        "skin_flaking", "hair_loss", "nodal_skin_eruptions",
    ],
    "Athletes Foot": [
        "itching", "skin_peeling", "skin_cracking", "skin_burning",
        "skin_scaling", "skin_rash", "foot_pain", "skin_fissures",
        "oozing_from_skin",
    ],
    "Vitiligo": [
        "dischromic_patches", "skin_lesions", "hair_loss",
        "photosensitivity", "dry_skin",
    ],
    "Seborrheic Dermatitis": [
        "skin_flaking", "skin_scaling", "itching", "oily_skin",
        "skin_rash", "facial_redness", "scalp_tenderness",
        "crust_formation",
    ],

    # ── GASTROINTESTINAL DISEASES ────────────────────────────────
    "GERD": [
        "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting",
        "cough", "chest_pain", "heartburn", "acid_reflux",
        "regurgitation", "difficulty_swallowing",
    ],
    "Chronic cholestasis": [
        "itching", "vomiting", "yellowish_skin", "nausea",
        "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes",
        "clay_colored_stool", "dark_urine",
    ],
    "Peptic ulcer diseae": [
        "vomiting", "loss_of_appetite", "abdominal_pain",
        "passage_of_gases", "internal_itching", "epigastric_pain",
        "heartburn", "nausea", "blood_in_vomit",
    ],
    "Gastroenteritis": [
        "vomiting", "sunken_eyes", "dehydration", "diarrhoea",
        "abdominal_cramping", "nausea", "mild_fever",
        "loss_of_appetite", "body_aches",
    ],
    "Jaundice": [
        "itching", "vomiting", "fatigue", "weight_loss", "high_fever",
        "yellowish_skin", "dark_urine", "abdominal_pain",
        "yellowing_of_eyes", "clay_colored_stool", "nausea",
    ],
    "hepatitis A": [
        "joint_pain", "vomiting", "yellowish_skin", "dark_urine",
        "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea",
        "mild_fever", "yellowing_of_eyes", "muscle_pain", "fatigue",
    ],
    "Hepatitis B": [
        "itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine",
        "loss_of_appetite", "abdominal_pain", "yellow_urine",
        "yellowing_of_eyes", "malaise", "receiving_blood_transfusion",
        "receiving_unsterile_injections",
    ],
    "Hepatitis C": [
        "fatigue", "yellowish_skin", "nausea", "loss_of_appetite",
        "yellowing_of_eyes", "family_history", "joint_pain",
        "muscle_pain",
    ],
    "Hepatitis D": [
        "joint_pain", "vomiting", "fatigue", "yellowish_skin",
        "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain",
        "yellowing_of_eyes",
    ],
    "Hepatitis E": [
        "joint_pain", "vomiting", "fatigue", "high_fever",
        "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite",
        "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure",
        "coma", "stomach_bleeding",
    ],
    "Alcoholic hepatitis": [
        "vomiting", "yellowish_skin", "abdominal_pain",
        "swelling_of_stomach", "distention_of_abdomen",
        "history_of_alcohol_consumption", "fluid_overload",
        "nausea", "fatigue",
    ],
    "Irritable Bowel Syndrome": [
        "abdominal_pain", "abdominal_cramping", "abdominal_bloating",
        "diarrhoea", "constipation", "excessive_gas",
        "passage_of_gases", "mucus_in_stool", "nausea",
        "feeling_full_quickly",
    ],
    "Appendicitis": [
        "right_lower_abdominal_pain", "abdominal_pain", "nausea",
        "vomiting", "high_fever", "loss_of_appetite",
        "abdominal_cramping", "constipation", "diarrhoea",
    ],
    "Gallstones": [
        "right_upper_abdominal_pain", "abdominal_pain", "nausea",
        "vomiting", "yellowish_skin", "clay_colored_stool",
        "dark_urine", "indigestion", "heartburn",
    ],
    "Pancreatitis": [
        "epigastric_pain", "abdominal_pain", "nausea", "vomiting",
        "high_fever", "fast_heart_rate", "weight_loss",
        "back_pain", "abdominal_bloating", "greasy_stool",
    ],
    "Celiac Disease": [
        "diarrhoea", "abdominal_bloating", "weight_loss", "fatigue",
        "abdominal_pain", "greasy_stool", "nausea", "skin_rash",
        "joint_pain", "bone_pain",
    ],
    "Food Poisoning": [
        "vomiting", "nausea", "diarrhoea", "abdominal_cramping",
        "high_fever", "dehydration", "body_aches", "headache",
        "weakness_in_limbs", "blood_in_vomit",
    ],
    "Gastritis": [
        "stomach_pain", "nausea", "vomiting", "indigestion",
        "loss_of_appetite", "abdominal_bloating", "heartburn",
        "blood_in_vomit", "feeling_full_quickly", "epigastric_pain",
    ],
    "Dimorphic hemmorhoids(piles)": [
        "constipation", "pain_during_bowel_movements",
        "pain_in_anal_region", "bloody_stool",
        "irritation_in_anus", "rectal_bleeding",
        "rectal_pain",
    ],

    # ── KIDNEY / URINARY ─────────────────────────────────────────
    "Urinary tract infection": [
        "burning_micturition", "bladder_discomfort",
        "foul_smell_of_urine", "continuous_feel_of_urine",
        "urinary_urgency", "pelvic_pain", "cloudy_urine",
        "blood_in_urine", "mild_fever",
    ],
    "Kidney Stones": [
        "flank_pain", "kidney_pain", "groin_pain", "blood_in_urine",
        "nausea", "vomiting", "burning_micturition",
        "urinary_urgency", "restlessness", "sweating",
    ],
    "Chronic Kidney Disease": [
        "fatigue", "swollen_legs", "edema", "nausea",
        "loss_of_appetite", "decreased_urine_output", "foamy_urine",
        "itching", "muscle_cramps", "breathlessness",
        "puffy_face_and_eyes", "metallic_taste",
    ],
    "Pyelonephritis": [
        "high_fever", "flank_pain", "kidney_pain",
        "burning_micturition", "nausea", "vomiting",
        "cloudy_urine", "blood_in_urine", "chills",
        "frequent_infections", "back_pain",
    ],
    "Nephrotic Syndrome": [
        "edema", "swollen_legs", "foamy_urine", "fatigue",
        "loss_of_appetite", "weight_gain", "puffy_face_and_eyes",
        "abdominal_bloating",
    ],
    "Bladder Infection": [
        "burning_micturition", "urinary_urgency",
        "continuous_feel_of_urine", "pelvic_pain",
        "cloudy_urine", "blood_in_urine", "foul_smell_of_urine",
        "mild_fever", "abdominal_cramping",
    ],
    "Prostatitis": [
        "burning_micturition", "difficulty_starting_urination",
        "weak_urine_stream", "pelvic_pain", "groin_pain",
        "testicular_pain", "high_fever", "chills",
        "nighttime_urination", "lower_back_pain",
    ],

    # ── ENDOCRINE / METABOLIC ────────────────────────────────────
    "Diabetes": [
        "fatigue", "weight_loss", "restlessness", "lethargy",
        "irregular_sugar_level", "blurred_and_distorted_vision",
        "obesity", "excessive_hunger", "increased_appetite",
        "polyuria", "excessive_thirst", "slow_wound_healing",
        "frequent_infections",
    ],
    "Hypothyroidism": [
        "fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings",
        "lethargy", "dizziness", "puffy_face_and_eyes",
        "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
        "depression", "irritability", "abnormal_menstruation",
        "dry_skin", "hair_loss", "cold_intolerance",
        "constipation", "muscle_cramps",
    ],
    "Hyperthyroidism": [
        "fatigue", "mood_swings", "weight_loss", "restlessness",
        "sweating", "diarrhoea", "fast_heart_rate",
        "excessive_hunger", "muscle_weakness", "irritability",
        "abnormal_menstruation", "heat_intolerance", "tremors",
        "anxiety", "palpitations", "hair_loss",
    ],
    "Hypoglycemia": [
        "vomiting", "fatigue", "anxiety", "sweating", "headache",
        "nausea", "blurred_and_distorted_vision", "excessive_hunger",
        "drying_and_tingling_lips", "slurred_speech", "irritability",
        "palpitations", "tremors", "confusion_episodes", "fainting",
    ],
    "Cushing Syndrome": [
        "weight_gain", "obesity", "flushing", "fatigue",
        "muscle_weakness", "easy_bruising",
        "mood_changes", "high_fever", "irregular_periods",
        "slow_wound_healing",
    ],
    "Addison Disease": [
        "fatigue", "chronic_fatigue", "weight_loss", "muscle_weakness",
        "loss_of_appetite", "nausea", "vomiting",
        "abdominal_pain", "dizziness", "fainting",
        "depression", "irritability",
    ],
    "PCOS": [
        "irregular_periods", "weight_gain",
        "hair_loss", "thinning_hair", "fatigue",
        "mood_swings", "pelvic_pain", "depression",
        "oily_skin", "skin_bumps",
    ],
    "Gout": [
        "joint_pain", "joint_redness", "joint_warmth",
        "swelling_joints", "foot_pain", "knee_pain",
        "limited_range_of_motion", "high_fever",
    ],
    "Vitamin D Deficiency": [
        "bone_pain", "muscle_weakness", "fatigue",
        "depression", "slow_wound_healing", "hair_loss",
        "muscle_pain", "back_pain", "mood_changes",
        "frequent_infections",
    ],

    # ── INFECTIOUS DISEASES ──────────────────────────────────────
    "Malaria": [
        "chills", "vomiting", "high_fever", "sweating", "headache",
        "nausea", "diarrhoea", "muscle_pain", "fatigue",
        "body_aches", "shivering",
    ],
    "Dengue": [
        "skin_rash", "chills", "joint_pain", "vomiting", "fatigue",
        "high_fever", "headache", "nausea", "loss_of_appetite",
        "pain_behind_the_eyes", "back_pain", "malaise",
        "muscle_pain", "red_spots_over_body", "body_aches",
    ],
    "Typhoid": [
        "chills", "vomiting", "fatigue", "high_fever", "headache",
        "nausea", "constipation", "abdominal_pain", "diarrhoea",
        "toxic_look_(typhos)", "belly_pain", "loss_of_appetite",
    ],
    "AIDS": [
        "muscle_wasting", "patches_in_throat", "high_fever",
        "extra_marital_contacts", "weight_loss", "chronic_fatigue",
        "night_sweats", "swelled_lymph_nodes", "diarrhoea",
        "frequent_infections", "skin_rash",
    ],
    "Chicken pox": [
        "itching", "skin_rash", "fatigue", "lethargy", "high_fever",
        "headache", "loss_of_appetite", "mild_fever",
        "swelled_lymph_nodes", "malaise", "red_spots_over_body",
        "blister", "vesicles",
    ],
    "Allergy": [
        "continuous_sneezing", "shivering", "chills",
        "watering_from_eyes", "eye_itching", "congestion",
        "skin_rash", "hives", "itching",
    ],
    "Drug Reaction": [
        "itching", "skin_rash", "stomach_pain",
        "burning_micturition", "spotting_urination",
        "high_fever", "nausea", "skin_bumps",
    ],
    "Measles": [
        "high_fever", "cough", "runny_nose", "redness_of_eyes",
        "skin_rash", "red_spots_over_body", "sore_throat",
        "fatigue", "loss_of_appetite", "sensitivity_to_light",
        "white_patches_on_tongue",
    ],
    "Mumps": [
        "high_fever", "facial_swelling", "jaw_pain", "swollen_face",
        "headache", "fatigue", "loss_of_appetite", "muscle_pain",
        "painful_swallowing", "ear_pain", "jaw_stiffness",
    ],
    "Rubella": [
        "mild_fever", "skin_rash", "swelled_lymph_nodes", "joint_pain",
        "headache", "redness_of_eyes", "congestion", "fatigue",
        "runny_nose",
    ],
    "Mononucleosis": [
        "fatigue", "chronic_fatigue", "sore_throat", "high_fever",
        "swelled_lymph_nodes", "headache", "skin_rash",
        "muscle_pain", "swollen_face", "loss_of_appetite",
        "night_sweats",
    ],
    "Lyme Disease": [
        "ring_shaped_rash", "fatigue", "joint_pain", "headache",
        "high_fever", "chills", "muscle_pain", "stiff_neck",
        "swelled_lymph_nodes", "facial_drooping", "memory_loss",
    ],
    "Tetanus": [
        "lockjaw", "jaw_stiffness", "muscle_spasms",
        "difficulty_swallowing", "high_fever", "sweating",
        "stiff_neck", "muscle_pain", "fast_heart_rate",
    ],
    "Cholera": [
        "diarrhoea", "vomiting", "dehydration", "muscle_cramps",
        "nausea", "sunken_eyes", "dry_mouth", "excessive_thirst",
        "lethargy", "low_energy",
    ],
    "Hand Foot Mouth Disease": [
        "high_fever", "sore_throat", "mouth_sores", "skin_rash",
        "blister", "loss_of_appetite", "painful_swallowing",
        "malaise", "irritability",
    ],
    "Scarlet Fever": [
        "high_fever", "sore_throat", "skin_rash", "red_spots_over_body",
        "tongue_swelling", "headache", "nausea", "vomiting",
        "body_aches", "swelled_lymph_nodes", "flushing",
    ],
    "Diphtheria": [
        "sore_throat", "high_fever", "swelled_lymph_nodes",
        "throat_swelling", "difficulty_swallowing", "hoarseness",
        "patches_in_throat", "malaise", "skin_lesions",
    ],
    "Chikungunya": [
        "high_fever", "joint_pain", "skin_rash", "headache",
        "muscle_pain", "fatigue", "nausea", "swelling_joints",
        "redness_of_eyes",
    ],
    "Zika Virus": [
        "mild_fever", "skin_rash", "joint_pain", "redness_of_eyes",
        "headache", "muscle_pain", "fatigue", "eye_pain",
    ],
    "Viral Fever": [
        "high_fever", "body_aches", "headache", "fatigue",
        "muscle_pain", "chills", "sweating", "loss_of_appetite",
        "sore_throat", "runny_nose", "mild_fever",
    ],

    # ── ENT ──────────────────────────────────────────────────────
    "Otitis Media": [
        "ear_pain", "ear_discharge", "hearing_loss", "high_fever",
        "irritability", "headache", "ear_fullness",
        "balance_problems", "muffled_hearing",
    ],
    "Tonsillitis": [
        "sore_throat", "painful_swallowing", "high_fever",
        "swelled_lymph_nodes", "headache", "throat_swelling",
        "white_patches_on_tongue", "bad_breath", "ear_pain",
        "hoarseness", "stiff_neck",
    ],
    "Meniere Disease": [
        "spinning_movements", "tinnitus_ringing", "hearing_loss",
        "ear_fullness", "nausea", "vomiting",
        "balance_problems", "unsteadiness",
    ],
    "Tinnitus": [
        "tinnitus_ringing", "hearing_loss", "ear_fullness",
        "headache", "dizziness", "anxiety", "sleep_disturbance",
        "poor_concentration",
    ],
    "Nasal Polyps": [
        "congestion", "nasal_obstruction", "runny_nose",
        "post_nasal_drip", "loss_of_smell", "loss_of_taste",
        "facial_pressure", "headache", "snoring",
    ],
    "Deviated Septum": [
        "nasal_obstruction", "congestion", "nosebleeds",
        "facial_pressure", "headache", "snoring",
        "post_nasal_drip", "sinus_pressure",
    ],
    "Vestibular Neuritis": [
        "spinning_movements", "nausea", "vomiting",
        "balance_problems", "unsteadiness", "dizziness",
    ],
    "Labyrinthitis": [
        "spinning_movements", "hearing_loss", "tinnitus_ringing",
        "nausea", "vomiting", "balance_problems",
        "unsteadiness", "ear_pain", "ear_fullness",
    ],
    "Epiglottitis": [
        "sore_throat", "painful_swallowing", "high_fever",
        "difficulty_swallowing", "breathlessness", "stridor",
        "drooling", "hoarseness", "throat_swelling",
    ],
    "Peritonsillar Abscess": [
        "sore_throat", "painful_swallowing", "high_fever",
        "swelled_lymph_nodes", "throat_swelling", "ear_pain",
        "drooling", "hoarseness", "bad_breath", "lockjaw",
    ],

    # ── EYE / OPHTHALMOLOGY ──────────────────────────────────────
    "Conjunctivitis": [
        "red_eye", "eye_discharge", "eye_itching",
        "watering_from_eyes", "redness_of_eyes",
        "crusty_eyelids", "photophobia", "eye_burning",
        "foreign_body_sensation_eye",
    ],
    "Glaucoma": [
        "eye_pain", "blurred_and_distorted_vision", "headache",
        "nausea", "vomiting", "redness_of_eyes",
        "vision_loss", "light_flashes_in_vision",
        "photophobia",
    ],
    "Stye": [
        "swollen_eyelid", "eye_pain", "eye_swelling",
        "red_eye", "crusty_eyelids", "watering_from_eyes",
        "photophobia", "foreign_body_sensation_eye",
    ],
    "Dry Eye Syndrome": [
        "dry_eyes", "eye_burning", "eye_itching",
        "blurred_and_distorted_vision", "redness_of_eyes",
        "photophobia", "foreign_body_sensation_eye",
        "watering_from_eyes", "fatigue",
    ],
    "Uveitis": [
        "eye_pain", "red_eye", "blurred_and_distorted_vision",
        "photophobia", "floaters_in_vision",
        "vision_loss", "headache", "redness_of_eyes",
    ],
    "Blepharitis": [
        "crusty_eyelids", "eye_itching", "swollen_eyelid",
        "eye_burning", "redness_of_eyes", "dry_eyes",
        "watering_from_eyes", "photophobia",
    ],
    "Corneal Abrasion": [
        "eye_pain", "watering_from_eyes", "red_eye",
        "photophobia", "blurred_and_distorted_vision",
        "foreign_body_sensation_eye", "headache",
    ],
    "Orbital Cellulitis": [
        "eye_swelling", "eye_pain", "red_eye", "high_fever",
        "blurred_and_distorted_vision", "double_vision",
        "headache", "swollen_eyelid", "vision_loss",
    ],

    # ── AUTOIMMUNE / RHEUMATOLOGY ────────────────────────────────
    "Osteoarthristis": [
        "joint_pain", "neck_pain", "knee_pain", "hip_joint_pain",
        "swelling_joints", "painful_walking", "morning_stiffness",
        "limited_range_of_motion", "bone_pain",
    ],
    "Rheumatoid Arthritis": [
        "joint_pain", "swelling_joints", "morning_stiffness",
        "fatigue", "muscle_weakness", "stiff_neck",
        "movement_stiffness", "painful_walking",
        "joint_warmth", "joint_redness", "finger_swelling",
    ],
    "Lupus": [
        "butterfly_rash", "joint_pain", "fatigue", "high_fever",
        "photosensitivity", "hair_loss", "mouth_ulcers_recurrent",
        "chest_pain", "breathlessness", "raynaud_phenomenon",
        "swelling_joints", "skin_rash", "kidney_pain",
    ],
    "Multiple Sclerosis": [
        "numbness", "tingling", "muscle_weakness",
        "blurred_and_distorted_vision", "fatigue",
        "coordination_problems", "slurred_speech",
        "tremors", "electric_shock_sensation",
        "progressive_weakness", "balance_problems",
        "bladder_discomfort",
    ],
    "Sjogren Syndrome": [
        "dry_eyes", "dry_mouth", "dry_eyes_and_mouth", "fatigue",
        "joint_pain", "skin_rash", "dry_skin",
    ],
    "Ankylosing Spondylitis": [
        "lower_back_pain", "back_pain", "morning_stiffness",
        "stiff_neck", "fatigue", "hip_joint_pain",
        "movement_stiffness", "chest_tightness",
        "breathlessness",
    ],
    "Psoriatic Arthritis": [
        "joint_pain", "swelling_joints", "skin_rash",
        "skin_peeling", "silver_like_dusting", "nail_discoloration",
        "nail_thickening", "morning_stiffness", "fatigue",
        "finger_swelling", "lower_back_pain",
    ],
    "Fibromyalgia": [
        "muscle_pain", "chronic_fatigue", "fatigue",
        "sleep_disturbance", "brain_fog", "headache",
        "depression", "anxiety", "morning_stiffness",
        "tingling", "abdominal_pain", "irritability",
    ],

    # ── GENERAL / OTHER ──────────────────────────────────────────
    "Anemia": [
        "fatigue", "pale_skin", "breathlessness", "dizziness",
        "cold_hands_and_feets", "headache", "fast_heart_rate",
        "brittle_nails", "chest_pain", "general_weakness",
        "palpitations",
    ],
    "Iron Deficiency": [
        "fatigue", "pale_skin", "brittle_nails", "hair_loss",
        "dizziness", "cold_hands_and_feets", "headache",
        "sore_throat", "dry_skin", "mouth_sores",
    ],
    "Heat Stroke": [
        "high_fever", "nausea", "vomiting",
        "headache", "confusion_episodes", "fast_heart_rate",
        "breathlessness", "muscle_cramps", "fainting",
        "seizures",
    ],
    "Dehydration Syndrome": [
        "dehydration", "excessive_thirst", "dry_mouth",
        "sunken_eyes", "fatigue", "dizziness",
        "decreased_urine_output", "dark_urine",
        "headache", "muscle_cramps",
    ],
    "Chronic Fatigue Syndrome": [
        "chronic_fatigue", "fatigue", "sleep_disturbance",
        "muscle_pain", "joint_pain", "headache",
        "sore_throat", "swelled_lymph_nodes",
        "poor_concentration", "memory_loss",
        "brain_fog", "depression",
    ],
    "Dimorphic hemmorhoids(piles)": [
        "constipation", "pain_during_bowel_movements",
        "pain_in_anal_region", "bloody_stool",
        "irritation_in_anus", "rectal_bleeding",
        "rectal_pain",
    ],
}

SAMPLES_PER_DISEASE = 150


def generate_samples(disease_name, symptom_list, n_samples, all_symptoms):
    """Generate n_samples rows for a disease with realistic variation."""
    rows = []
    n_symp = len(symptom_list)
    for _ in range(n_samples):
        vec = {s: 0 for s in all_symptoms}
        keep = max(2, int(n_symp * random.uniform(0.7, 1.0)))
        active = random.sample(symptom_list, min(keep, n_symp))
        for s in active:
            vec[s] = 1
        vec["prognosis"] = disease_name
        rows.append(vec)
    return rows


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    symptom_set = set(SYMPTOMS)
    for disease, syms in DISEASE_SYMPTOMS.items():
        bad = [s for s in syms if s not in symptom_set]
        if bad:
            for s in bad:
                SYMPTOMS.append(s)
                symptom_set.add(s)

    print(f"Symptom count : {len(SYMPTOMS)}")
    print(f"Disease count : {len(DISEASE_SYMPTOMS)}")

    train_rows = []
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        train_rows.extend(generate_samples(disease, symptoms, SAMPLES_PER_DISEASE, SYMPTOMS))

    train_df = pd.DataFrame(train_rows, columns=SYMPTOMS + ["prognosis"])
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_path = os.path.join(script_dir, "Training.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  -> {train_path}  ({len(train_df)} rows x {len(train_df.columns)} cols)")

    test_rows = []
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        vec = {s: 0 for s in SYMPTOMS}
        for s in symptoms:
            vec[s] = 1
        vec["prognosis"] = disease
        test_rows.append(vec)

    test_df = pd.DataFrame(test_rows, columns=SYMPTOMS + ["prognosis"])
    test_path = os.path.join(script_dir, "Testing.csv")
    test_df.to_csv(test_path, index=False)
    print(f"  -> {test_path}  ({len(test_df)} rows x {len(test_df.columns)} cols)")


if __name__ == "__main__":
    main()
