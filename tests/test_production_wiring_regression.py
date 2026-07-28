"""
Production Wiring Regression Test (tests/test_production_wiring_regression.py)
================================================================================
Automated CI regression gate:
Fails build if app.py, src/api.py, or models/predictor.py outputs drift
from MultiTurnDiagnosisEngine direct outputs.
"""

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine import MultiTurnDiagnosisEngine
from models.predictor import DiseasePredictor
from src.api import app


@pytest.fixture
def engine():
    return MultiTurnDiagnosisEngine()


@pytest.fixture
def predictor():
    return DiseasePredictor()


@pytest.fixture
def api_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predictor_matches_engine(engine, predictor):
    """Verifies DiseasePredictor outputs match engine.predict_initial exactly."""
    test_text = "high fever with cough and difficulty breathing"
    engine_res = engine.predict_initial(test_text)
    predictor_res = predictor.predict_from_text(test_text)

    assert predictor_res["top_prediction"]["disease"] == engine_res["primary_diagnosis"]
    assert predictor_res["candidate_diseases"] == engine_res["candidate_diseases"]
    assert predictor_res["next_question_symptom"] == engine_res["next_question_symptom"]


def test_api_predict_matches_engine(engine, api_client):
    """Verifies POST /predict API outputs match engine.predict_initial exactly."""
    test_text = "sudden crushing chest pain going to jaw with sweating"
    engine_res = engine.predict_initial(test_text)

    res = api_client.post("/predict", json={"symptoms_text": test_text})
    assert res.status_code == 200
    data = res.get_json()

    assert data["primary_diagnosis"] == engine_res["primary_diagnosis"]
    assert data["candidate_diseases"] == engine_res["candidate_diseases"]
    assert data["next_question_symptom"] == engine_res["next_question_symptom"]
