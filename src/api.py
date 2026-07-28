"""
REST API Layer (src/api.py)
===========================
Production REST API implementing stateful & endpoint-driven Multi-Turn Diagnosis:
  - POST /predict: Accepts initial natural language symptoms, returns top diagnosis & next clarifying question.
  - POST /predict/followup: Accepts follow-up YES/NO answer, returns updated differential & next question/diagnosis.
  - GET  /health: Health check endpoint.
  - GET  /model-info: Returns active model architecture & engine details.
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

from src.engine import MultiTurnDiagnosisEngine

# Setup Rotating File Logger
log_file = os.path.join(LOGS_DIR, "app.log")
handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger("SaarthiAPI")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = Flask(__name__)

# Initialize Multi-Turn Diagnosis Engine
try:
    engine = MultiTurnDiagnosisEngine()
    engine_name = "Verified Multi-Turn Information-Theoretic Engine (v7.0.0)"
    logger.info("MultiTurnDiagnosisEngine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load MultiTurnDiagnosisEngine: {e}")
    engine = None
    engine_name = "Error"


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    status = "healthy" if engine is not None else "unhealthy"
    return jsonify({
        "status": status,
        "active_engine": engine_name,
        "diseases_count": len(engine.canonical_diseases) if engine else 0,
        "symptoms_count": len(engine.symptom_cols) if engine else 0
    }), 200 if status == "healthy" else 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns active model details and metadata."""
    return jsonify({
        "model_name": engine_name,
        "engine_name": engine_name,
        "architecture": "Weighted Overlap Specificity + Information-Theoretic Entropy Question Selection",
        "symptom_count": len(engine.symptom_cols) if engine else 0,
        "disease_count": len(engine.canonical_diseases) if engine else 0,
        "hard_cap_followups": 2,
        "disclaimer": "Educational decision support tool only. Not clinically validated for medical diagnosis."
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Request JSON Body:
      {
        "symptoms_text": "high fever with cough and difficulty breathing"
      }
    """
    if not request.is_json:
        logger.warning("Request body must be JSON")
        return jsonify({"error": "Request content-type must be application/json"}), 400

    data = request.get_json()
    text = data.get("symptoms_text", "")

    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid symptoms_text provided")
        return jsonify({"error": "Field 'symptoms_text' is required and must be a non-empty string."}), 400

    logger.info(f"Received initial prediction request: '{text}'")

    try:
        result = engine.predict_initial(text)
        logger.info(f"Initial result for '{text}': {result['primary_diagnosis']} | Candidates: {len(result['candidate_diseases'])}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing prediction request: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/predict/followup", methods=["POST"])
def predict_followup():
    """
    POST /predict/followup
    Request JSON Body:
      {
        "present_symptoms": ["cough", "high_fever"],
        "absent_symptoms": [],
        "asked_symptoms": ["cough", "high_fever", "phlegm"],
        "user_answer": "YES",
        "current_question_symptom": "phlegm",
        "rounds_asked": 0
      }
    """
    if not request.is_json:
        return jsonify({"error": "Request content-type must be application/json"}), 400

    data = request.get_json()
    present_syms = data.get("present_symptoms", [])
    absent_syms = data.get("absent_symptoms", [])
    asked_syms = data.get("asked_symptoms", [])
    user_answer = data.get("user_answer", "YES")
    current_q_sym = data.get("current_question_symptom", None)
    rounds_asked = data.get("rounds_asked", 0)

    try:
        result = engine.process_followup(
            present_symptoms=present_syms,
            absent_symptoms=absent_syms,
            asked_symptoms=asked_syms,
            user_answer_yes_no=user_answer,
            current_question_symptom=current_q_sym,
            rounds_asked=rounds_asked
        )
        logger.info(f"Followup result (Round {result['rounds_asked']}): Top Diagnosis '{result['primary_diagnosis']}' | Candidates: {len(result['candidate_diseases'])}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing followup request: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
