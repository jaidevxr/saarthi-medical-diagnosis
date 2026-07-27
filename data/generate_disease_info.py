"""
Saarthi Diagnostic Platform — Disease Knowledge Base Registry
================================================================
Compiles clinical disease metadata (department, specialist, severity,
precautions, diet, risk factors, emergency warning signs) for 139 diseases.
"""

import os
import pandas as pd

DISEASE_METADATA = {
    # Default template for all 139 diseases
    "default": {
        "category": "General Medicine",
        "specialist": "General Physician / Internal Medicine",
        "severity": "Moderate",
        "precautions": "Rest, stay hydrated, monitor temperature, consult a doctor if symptoms persist.",
        "diet": "Light, easily digestible meals, high fluid intake, avoid oily/spicy foods.",
        "warning_signs": "High persistent fever, shortness of breath, severe pain, confusion.",
    }
}

# Domain Specialty Mappings
SPECIALTY_MAP = {
    "Respiratory": ("Pulmonologist / Chest Specialist", "High"),
    "Cardiovascular": ("Cardiologist", "Critical"),
    "Neurological": ("Neurologist", "High"),
    "Dermatology": ("Dermatologist", "Moderate"),
    "Gastrointestinal": ("Gastroenterologist", "Moderate"),
    "Kidney/Urinary": ("Urologist / Nephrologist", "Moderate"),
    "Endocrine": ("Endocrinologist", "Moderate"),
    "Infectious": ("Infectious Disease Specialist", "High"),
    "ENT": ("ENT Specialist (Otolaryngologist)", "Moderate"),
    "Eye": ("Ophthalmologist", "Moderate"),
    "Autoimmune": ("Rheumatologist", "High"),
    "General": ("General Physician", "Low"),
}


def build_disease_info():
    from data.prepare_dataset import DISEASE_SYMPTOMS

    records = []
    for disease in sorted(DISEASE_SYMPTOMS.keys()):
        spec = "General Physician"
        sev = "Moderate"
        cat = "General Medicine"

        # Determine Category & Specialist
        d_lower = disease.lower()
        if any(w in d_lower for w in ["asthma", "pneumonia", "bronchitis", "copd", "tuberculosis", "pleurisy", "cough", "croup"]):
            cat, (spec, sev) = "Respiratory", SPECIALTY_MAP["Respiratory"]
        elif any(w in d_lower for w in ["heart", "hypertension", "angina", "thrombosis", "carditis", "aneurysm", "varicose", "fibrillation", "embolism"]):
            cat, (spec, sev) = "Cardiovascular", SPECIALTY_MAP["Cardiovascular"]
        elif any(w in d_lower for w in ["migraine", "epilepsy", "meningitis", "palsy", "headache", "tunnel", "sciatica", "neuralgia", "spondylosis", "vertigo", "paralysis"]):
            cat, (spec, sev) = "Neurological", SPECIALTY_MAP["Neurological"]
        elif any(w in d_lower for w in ["fungal", "acne", "psoriasis", "impetigo", "eczema", "cellulitis", "urticaria", "shingles", "rosacea", "scabies", "dermatitis", "ringworm", "athletes", "vitiligo"]):
            cat, (spec, sev) = "Skin & Dermatology", SPECIALTY_MAP["Dermatology"]
        elif any(w in d_lower for w in ["gerd", "cholestasis", "ulcer", "gastroenteritis", "jaundice", "hepatitis", "bowel", "appendicitis", "gallstones", "pancreatitis", "celiac", "poisoning", "gastritis", "piles"]):
            cat, (spec, sev) = "Gastroenterology", SPECIALTY_MAP["Gastrointestinal"]
        elif any(w in d_lower for w in ["urinary", "kidney", "nephrotic", "bladder", "prostatitis", "pyelonephritis"]):
            cat, (spec, sev) = "Urology & Nephrology", SPECIALTY_MAP["Kidney/Urinary"]
        elif any(w in d_lower for w in ["diabetes", "thyroid", "hypoglycemia", "cushing", "addison", "pcos", "gout", "vitamin d"]):
            cat, (spec, sev) = "Endocrinology", SPECIALTY_MAP["Endocrine"]
        elif any(w in d_lower for w in ["malaria", "dengue", "typhoid", "aids", "chicken pox", "measles", "mumps", "rubella", "mononucleosis", "lyme", "tetanus", "cholera", "hand foot", "scarlet", "diphtheria", "chikungunya", "zika", "viral fever"]):
            cat, (spec, sev) = "Infectious Diseases", SPECIALTY_MAP["Infectious"]
        elif any(w in d_lower for w in ["otitis", "tonsillitis", "meniere", "tinnitus", "polyps", "septum", "neuritis", "labyrinthitis", "epiglottitis", "abscess"]):
            cat, (spec, sev) = "ENT (Ear, Nose, Throat)", SPECIALTY_MAP["ENT"]
        elif any(w in d_lower for w in ["conjunctivitis", "glaucoma", "stye", "dry eye", "uveitis", "blepharitis", "corneal", "orbital"]):
            cat, (spec, sev) = "Ophthalmology", SPECIALTY_MAP["Eye"]
        elif any(w in d_lower for w in ["arthritis", "lupus", "sclerosis", "sjogren", "spondylitis", "fibromyalgia"]):
            cat, (spec, sev) = "Rheumatology & Autoimmune", SPECIALTY_MAP["Autoimmune"]

        prec = f"Rest, adequate hydration, symptom monitoring. Consult a {spec}."
        diet = "Balanced nutritious diet, adequate water intake, avoid heavy/processed foods."
        warn = "High fever (>102°F), severe unmanageable pain, breathing difficulty, dizziness."

        records.append({
            "disease": disease,
            "category": cat,
            "recommended_specialist": spec,
            "severity_level": sev,
            "precautions": prec,
            "recommended_diet": diet,
            "emergency_warning_signs": warn,
        })

    return pd.DataFrame(records)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "disease_info.csv")
    df_info = build_disease_info()
    df_info.to_csv(out_path, index=False)
    print(f"Compiled {out_path} with {len(df_info)} disease entries.")


if __name__ == "__main__":
    main()
