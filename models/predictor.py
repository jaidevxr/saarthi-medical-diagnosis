"""
Disease Predictor Module
========================
Provides high-level OOP API for loading models and generating disease predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np

from utils.helpers import parse_symptoms_with_metadata

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


class DiseasePredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.encoder = joblib.load(os.path.join(models_dir, "encoder.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.symptom_columns = joblib.load(os.path.join(models_dir, "symptom_columns.pkl"))

        # Load trained models if available
        self.models = {}
        for m_name in ["calibrated_nb", "random_forest_tuned", "extra_trees", "random_forest", "naive_bayes"]:
            p = os.path.join(models_dir, f"{m_name}.pkl")
            if os.path.exists(p):
                self.models[m_name] = joblib.load(p)

        # Primary model is Calibrated Naive Bayes or Random Forest Tuned
        if "calibrated_nb" in self.models:
            self.primary_model = self.models["calibrated_nb"]
        elif "random_forest_tuned" in self.models:
            self.primary_model = self.models["random_forest_tuned"]
        else:
            self.primary_model = list(self.models.values())[0] if self.models else None

    def predict_from_text(self, text, top_n=5):
        """
        Parses text input for symptoms, builds feature vector, and predicts top diseases.
        """
        meta = parse_symptoms_with_metadata(text, self.symptom_columns)
        detected = meta["matched"]

        feat = pd.DataFrame([[0] * len(self.symptom_columns)], columns=self.symptom_columns)
        for s in detected:
            if s in feat.columns:
                feat[s] = 1

        feat_sc = self.scaler.transform(feat)
        probs = self.primary_model.predict_proba(feat_sc)[0]

        top_indices = np.argsort(probs)[-top_n:][::-1]
        diseases = self.encoder.inverse_transform(top_indices)

        top_predictions = []
        for d, idx in zip(diseases, top_indices):
            top_predictions.append({
                "disease": d,
                "confidence": round(float(probs[idx]) * 100, 2)
            })

        return {
            "top_prediction": top_predictions[0] if top_predictions else None,
            "top_3": top_predictions[:3],
            "top_5": top_predictions[:5],
            "symptoms_matched": detected,
            "symptoms_unmatched": meta.get("unmatched", [])
        }
