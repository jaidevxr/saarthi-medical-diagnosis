"""
Integration Tests for REST API Layer (tests/test_api.py)
"""

import pytest
import json
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_model_info_endpoint(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert "model_name" in data
    assert "disclaimer" in data

def test_predict_endpoint_valid_input(client):
    payload = {"symptoms_text": "high fever with cough and difficulty breathing"}
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "primary_diagnosis" in data
    assert "confidence" in data
    assert "matched_symptoms" in data
    assert "explanation" in data

def test_predict_endpoint_empty_input(client):
    payload = {"symptoms_text": ""}
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 400
