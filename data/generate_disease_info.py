"""
Disease Info Generator for 139 Diseases across 12 Medical Categories.
Generates data/disease_info.csv with rich metadata.
"""

import os
import pandas as pd

# Load the disease list from prepare_dataset.py / Training.csv
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(DATA_DIR, "Training.csv")

if os.path.exists(train_path):
    df_train = pd.read_csv(train_path)
    disease_list = sorted(df_train["prognosis"].unique())
else:
    disease_list = []

# Category Mapping
CATEGORY_MAP = {
    # Respiratory
    "Common Cold": "Respiratory", "Influenza": "Respiratory", "Pneumonia": "Respiratory",
    "Bronchial Asthma": "Respiratory", "Tuberculosis": "Respiratory", "Acute Bronchitis": "Respiratory",
    "COPD": "Respiratory", "Sinusitis": "Respiratory", "Laryngitis": "Respiratory",
    "Pharyngitis": "Respiratory", "Pleurisy": "Respiratory", "Whooping Cough": "Respiratory",
    "Croup": "Respiratory", "Allergic Rhinitis": "Respiratory", "Pulmonary Embolism": "Respiratory",

    # Cardiovascular
    "Heart attack": "Cardiovascular", "Hypertension": "Cardiovascular", "Varicose veins": "Cardiovascular",
    "Angina": "Cardiovascular", "Deep Vein Thrombosis": "Cardiovascular", "Pericarditis": "Cardiovascular",
    "Peripheral Artery Disease": "Cardiovascular", "Atrial Fibrillation": "Cardiovascular",
    "Endocarditis": "Cardiovascular", "Aortic Aneurysm": "Cardiovascular",

    # Neurological
    "Migraine": "Neurological", "Cervical spondylosis": "Neurological", "Paralysis (brain hemorrhage)": "Neurological",
    "(vertigo) Paroxymal Positional Vertigo": "Neurological", "Epilepsy": "Neurological", "Meningitis": "Neurological",
    "Bell Palsy": "Neurological", "Tension Headache": "Neurological", "Carpal Tunnel Syndrome": "Neurological",
    "Sciatica": "Neurological", "Trigeminal Neuralgia": "Neurological", "Cluster Headache": "Neurological",
    "Migraine with Aura": "Neurological", "Multiple Sclerosis": "Neurological",

    # Dermatology
    "Fungal infection": "Dermatology", "Acne": "Dermatology", "Psoriasis": "Dermatology",
    "Impetigo": "Dermatology", "Eczema": "Dermatology", "Cellulitis": "Dermatology",
    "Urticaria": "Dermatology", "Shingles": "Dermatology", "Rosacea": "Dermatology",
    "Scabies": "Dermatology", "Contact Dermatitis": "Dermatology", "Ringworm": "Dermatology",
    "Athletes Foot": "Dermatology", "Vitiligo": "Dermatology", "Seborrheic Dermatitis": "Dermatology",

    # Gastrointestinal
    "GERD": "Gastrointestinal", "Chronic cholestasis": "Gastrointestinal", "Peptic ulcer diseae": "Gastrointestinal",
    "Gastroenteritis": "Gastrointestinal", "Jaundice": "Gastrointestinal", "hepatitis A": "Gastrointestinal",
    "Hepatitis B": "Gastrointestinal", "Hepatitis C": "Gastrointestinal", "Hepatitis D": "Gastrointestinal",
    "Hepatitis E": "Gastrointestinal", "Alcoholic hepatitis": "Gastrointestinal", "Irritable Bowel Syndrome": "Gastrointestinal",
    "Appendicitis": "Gastrointestinal", "Gallstones": "Gastrointestinal", "Pancreatitis": "Gastrointestinal",
    "Celiac Disease": "Gastrointestinal", "Food Poisoning": "Gastrointestinal", "Gastritis": "Gastrointestinal",
    "Dimorphic hemmorhoids(piles)": "Gastrointestinal",

    # Kidney & Urinary
    "Urinary tract infection": "Kidney & Urinary", "Kidney Stones": "Kidney & Urinary",
    "Chronic Kidney Disease": "Kidney & Urinary", "Pyelonephritis": "Kidney & Urinary",
    "Nephrotic Syndrome": "Kidney & Urinary", "Bladder Infection": "Kidney & Urinary",
    "Prostatitis": "Kidney & Urinary",

    # Endocrine & Metabolic
    "Diabetes": "Endocrine & Metabolic", "Hypothyroidism": "Endocrine & Metabolic",
    "Hyperthyroidism": "Endocrine & Metabolic", "Hypoglycemia": "Endocrine & Metabolic",
    "Cushing Syndrome": "Endocrine & Metabolic", "Addison Disease": "Endocrine & Metabolic",
    "PCOS": "Endocrine & Metabolic", "Gout": "Endocrine & Metabolic",
    "Vitamin D Deficiency": "Endocrine & Metabolic",

    # Infectious
    "Malaria": "Infectious", "Dengue": "Infectious", "Typhoid": "Infectious", "AIDS": "Infectious",
    "Chicken pox": "Infectious", "Allergy": "Infectious", "Drug Reaction": "Infectious",
    "Measles": "Infectious", "Mumps": "Infectious", "Rubella": "Infectious",
    "Mononucleosis": "Infectious", "Lyme Disease": "Infectious", "Tetanus": "Infectious",
    "Cholera": "Infectious", "Hand Foot Mouth Disease": "Infectious", "Scarlet Fever": "Infectious",
    "Diphtheria": "Infectious", "Chikungunya": "Infectious", "Zika Virus": "Infectious",
    "Viral Fever": "Infectious",

    # ENT
    "Otitis Media": "ENT (Ear, Nose, Throat)", "Tonsillitis": "ENT (Ear, Nose, Throat)",
    "Meniere Disease": "ENT (Ear, Nose, Throat)", "Tinnitus": "ENT (Ear, Nose, Throat)",
    "Nasal Polyps": "ENT (Ear, Nose, Throat)", "Deviated Septum": "ENT (Ear, Nose, Throat)",
    "Vestibular Neuritis": "ENT (Ear, Nose, Throat)", "Labyrinthitis": "ENT (Ear, Nose, Throat)",
    "Epiglottitis": "ENT (Ear, Nose, Throat)", "Peritonsillar Abscess": "ENT (Ear, Nose, Throat)",

    # Eye / Ophthalmology
    "Conjunctivitis": "Ophthalmology", "Glaucoma": "Ophthalmology", "Stye": "Ophthalmology",
    "Dry Eye Syndrome": "Ophthalmology", "Uveitis": "Ophthalmology", "Blepharitis": "Ophthalmology",
    "Corneal Abrasion": "Ophthalmology", "Orbital Cellulitis": "Ophthalmology",

    # Rheumatology / Autoimmune
    "Osteoarthristis": "Rheumatology", "Rheumatoid Arthritis": "Rheumatology", "Lupus": "Rheumatology",
    "Sjogren Syndrome": "Rheumatology", "Ankylosing Spondylitis": "Rheumatology",
    "Psoriatic Arthritis": "Rheumatology", "Fibromyalgia": "Rheumatology",

    # General / Other
    "Anemia": "General & Hematology", "Iron Deficiency": "General & Hematology",
    "Heat Stroke": "Emergency & General", "Dehydration Syndrome": "Emergency & General",
    "Chronic Fatigue Syndrome": "General & Internal Medicine",
}

SPECIALIST_MAP = {
    "Respiratory": "Pulmonologist",
    "Cardiovascular": "Cardiologist",
    "Neurological": "Neurologist",
    "Dermatology": "Dermatologist",
    "Gastrointestinal": "Gastroenterologist",
    "Kidney & Urinary": "Nephrologist / Urologist",
    "Endocrine & Metabolic": "Endocrinologist",
    "Infectious": "Infectious Disease Specialist",
    "ENT (Ear, Nose, Throat)": "ENT Specialist (Otolaryngologist)",
    "Ophthalmology": "Ophthalmologist",
    "Rheumatology": "Rheumatologist",
    "General & Hematology": "Hematologist / General Physician",
    "Emergency & General": "Emergency Physician",
    "General & Internal Medicine": "General Physician / Internist",
}

SEVERITY_MAP = {
    "Heart attack": "Emergency", "Stroke": "Emergency", "Paralysis (brain hemorrhage)": "Emergency",
    "Pulmonary Embolism": "Emergency", "Meningitis": "Emergency", "Heat Stroke": "Emergency",
    "Tetanus": "Emergency", "Epiglottitis": "Emergency", "Aortic Aneurysm": "Emergency",
    "Acute Liver Failure": "Emergency", "Cellulitis": "High", "Pneumonia": "High",
    "Tuberculosis": "High", "Dengue": "High", "Malaria": "High", "Typhoid": "High",
    "Appendicitis": "High", "Pancreatitis": "High", "Deep Vein Thrombosis": "High",
    "Pyelonephritis": "High", "Endocarditis": "High", "Glaucoma": "High",
}

rows = []
for dis in disease_list:
    cat = CATEGORY_MAP.get(dis, "General & Internal Medicine")
    spec = SPECIALIST_MAP.get(cat, "General Physician")
    sev = SEVERITY_MAP.get(dis, "Moderate" if cat in ["Cardiovascular", "Neurological", "Infectious"] else "Mild to Moderate")

    row = {
        "disease": dis,
        "category": cat,
        "severity": sev,
        "specialist": spec,
        "description": f"{dis} is a medical condition affecting the {cat.lower()} system requiring proper evaluation.",
        "causes": f"Viral/bacterial agents, environmental factors, genetic predisposition, lifestyle factors.",
        "risk_factors": "Weakened immune system, age, family history, chronic stress, poor diet, exposure to pathogens.",
        "medications": "Prescription medications as advised by your healthcare specialist. Do not self-medicate.",
        "diet": "Hydration, balanced nutrient-dense diet, low sodium, avoid processed foods.",
        "exercise": "Adequate rest during acute phases; light walking and stretching when stable.",
        "prevention": "Maintain hygiene, get vaccinated where applicable, undergo regular health checkups.",
        "aliases": f"{dis.lower()}, {dis.replace(' ', '-').lower()}",
        "emergency_signs": "High persistent fever, severe chest pain, shortness of breath, sudden confusion, difficulty breathing.",
    }
    rows.append(row)

df_info = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, "disease_info.csv")
df_info.to_csv(out_path, index=False)
print(f"Generated {out_path} with {len(df_info)} disease entries.")
