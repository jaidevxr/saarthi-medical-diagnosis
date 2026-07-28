"""
Pipeline Integration Test Suite
===============================
Tests end-to-end medical diagnosis prediction pipeline.
"""

import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predictor import DiseasePredictor
from src.utils.constants import MODELS_DIR

def test_disease_predictor_initialization():
    predictor = DiseasePredictor(models_dir=MODELS_DIR)
    assert predictor.primary_model is not None
    assert len(predictor.symptom_columns) == 397
    assert len(predictor.encoder.classes) == 168

def test_prediction_from_text():
    predictor = DiseasePredictor(models_dir=MODELS_DIR)
    query = "I have high fever, chills, and severe headache since yesterday."
    res = predictor.predict_from_text(query)
    assert res["top_prediction"] is not None
    assert len(res["top_3"]) == 3
    assert len(res["symptoms_matched"]) > 0

if __name__ == "__main__":
    test_disease_predictor_initialization()
    test_prediction_from_text()
    print("[OK] Pipeline tests passed cleanly.")
