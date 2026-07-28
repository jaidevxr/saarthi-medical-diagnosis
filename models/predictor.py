"""
Disease Predictor Module (OOP Interface — models/predictor.py)
==============================================================
Provides high-level OOP API for loading the production Multi-Turn Diagnosis Engine
and generating disease predictions with stateful follow-up capabilities.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine import MultiTurnDiagnosisEngine


class DiseasePredictor:
    """
    Production OOP class for multi-turn disease prediction.
    Interfaces with MultiTurnDiagnosisEngine for weighted overlap specificity matching
    and entropy-driven clarifying question selection.
    """

    def __init__(self, models_dir=None):
        self.engine = MultiTurnDiagnosisEngine()
        self.primary_model = self.engine
        self.symptom_columns = self.engine.symptom_cols
        class DummyEncoder:
            def __init__(self, classes):
                self.classes_ = classes
        self.encoder = DummyEncoder(self.engine.canonical_diseases)

    def predict_from_text(self, text, top_n=5):
        """
        Parses natural language text, extracts symptoms, computes weighted overlap scores,
        and returns primary diagnosis, differential, candidate set, and next clarifying question.
        """
        res = self.engine.predict_initial(text)
        
        top_predictions = []
        for d in res["top3_differential"][:top_n]:
            top_predictions.append({
                "disease": d,
                "confidence": 100.0 if d == res["primary_diagnosis"] else 50.0
            })

        return {
            "top_prediction": {"disease": res["primary_diagnosis"], "confidence": 100.0},
            "top_3": top_predictions[:3],
            "top_5": top_predictions[:top_n],
            "symptoms_matched": res["matched_symptoms"],
            "symptoms_unmatched": res["unmatched_tokens"],
            "candidate_diseases": res["candidate_diseases"],
            "next_question_symptom": res["next_question_symptom"],
            "next_question_text": res["next_question_text"],
            "is_resolved": res["is_resolved"]
        }
