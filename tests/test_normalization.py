"""
Unit Tests for Symptom Normalization Layer (tests/test_normalization.py)
"""

import pytest
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import SymptomNormalizer

@pytest.fixture
def normalizer():
    return SymptomNormalizer()

@pytest.fixture
def valid_symptoms():
    return [
        "high_fever", "mild_fever", "cough", "phlegm", "breathlessness",
        "headache", "stomach_pain", "chest_pain", "itching", "vomiting",
        "diarrhoea", "polyuria", "fast_heart_rate", "yellowish_skin",
        "dark_urine", "fatigue", "burning_micturition"
    ]

def test_exact_symptom_matching(normalizer, valid_symptoms):
    res = normalizer.extract_symptoms("I have high_fever and cough", valid_symptoms)
    assert "high_fever" in res["matched"]
    assert "cough" in res["matched"]

def test_typo_fuzzy_matching(normalizer, valid_symptoms):
    res = normalizer.extract_symptoms("high fevr with cogh and breathng problem", valid_symptoms)
    assert "high_fever" in res["matched"]
    assert "cough" in res["matched"]
    assert "breathlessness" in res["matched"]

def test_hinglish_symptom_matching(normalizer, valid_symptoms):
    res = normalizer.extract_symptoms("mujhe bukhar aur khansi hai", valid_symptoms)
    assert "high_fever" in res["matched"]
    assert "cough" in res["matched"]

def test_empty_text_input(normalizer, valid_symptoms):
    res = normalizer.extract_symptoms("", valid_symptoms)
    assert res["matched"] == []
    assert res["unmatched"] == []

def test_vague_query_normalization(normalizer, valid_symptoms):
    vague_phrases = [
        "I don't feel well today",
        "I don't feel good",
        "feeling sick today",
        "not feeling well",
        "unwell today",
        "feeling terrible"
    ]
    for phrase in vague_phrases:
        res = normalizer.extract_symptoms(phrase, valid_symptoms)
        assert res["matched"] == [], f"Vague phrase '{phrase}' should not match any specific symptom, but matched: {res['matched']}"
