"""
Generates Disease Severity Taxonomy Mapping (High / Medium / Low).
Saved to data/disease_severity.json
"""

import os, json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

HIGH_SEVERITY_DISEASES = {
    "Pneumonia", "Heart attack", "Tuberculosis", "Dengue", "Appendicitis",
    "Malaria", "Typhoid", "Acute Liver Failure", "Aortic Aneurysm", "Myocarditis",
    "Stroke", "Pulmonary Embolism", "Sepsis", "Meningitis", "Brain Tumor",
    "Kidney Failure", "Encephalitis", "Epiglottitis", "Peritonitis", "Pancreatitis",
    "Diabetic Ketoacidosis", "hepatitis A", "Hepatitis B", "Hepatitis C",
    "Hepatitis D", "Hepatitis E", "Chronic Cholestasis", "Internal Bleeding",
    "Severe Asthma", "Anaphylaxis", "Pneumothorax", "Coronary Artery Disease",
    "Heart Failure", "Deep Vein Thrombosis", "Spirometry / Lung Abscess",
    "Whooping Cough", "Intermittent Fever / Malaria"
}

LOW_SEVERITY_DISEASES = {
    "Acne", "Common Cold", "Fungal infection", "Allergy", "Tinea Versicolor",
    "Dermatitis", "Ringworm", "Callus", "Dry Skin", "Mild Gastritis",
    "Hiccups", "Motion Sickness"
}

def generate_severity_map():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    diseases = sorted(train_df["prognosis"].unique().tolist())
    
    severity_map = {}
    for d in diseases:
        d_clean = d.strip()
        if any(h.lower() in d_clean.lower() for h in HIGH_SEVERITY_DISEASES):
            severity_map[d] = "High"
        elif any(l.lower() in d_clean.lower() for l in LOW_SEVERITY_DISEASES):
            severity_map[d] = "Low"
        else:
            severity_map[d] = "Medium"

    out_path = os.path.join(DATA_DIR, "disease_severity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(severity_map, f, indent=2)

    high_cnt = sum(1 for v in severity_map.values() if v == "High")
    med_cnt = sum(1 for v in severity_map.values() if v == "Medium")
    low_cnt = sum(1 for v in severity_map.values() if v == "Low")

    print(f"[OK] Generated severity map for {len(severity_map)} diseases:")
    print(f"     High Risk: {high_cnt} | Medium Risk: {med_cnt} | Low Risk: {low_cnt}")
    print(f"     Saved to: {out_path}")

if __name__ == "__main__":
    generate_severity_map()
