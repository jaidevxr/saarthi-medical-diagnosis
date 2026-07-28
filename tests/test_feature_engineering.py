"""
Unit Tests for Feature Engineering Module (tests/test_feature_engineering.py)
"""

import pytest
import os, sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import FeatureExtractor

@pytest.fixture
def sample_data():
    cols = ["cough", "high_fever", "chest_pain", "breathlessness", "stomach_pain"]
    X = pd.DataFrame([
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ], columns=cols)
    y = pd.Series(["Pneumonia", "Common Cold", "Gastritis"])
    return X, y

def test_feature_extractor_fit_transform(sample_data):
    X, y = sample_data
    extractor = FeatureExtractor()
    extractor.fit(X, y)
    
    transformed = extractor.transform(X)
    assert "symptom_count" in transformed.columns
    assert "body_system_count" in transformed.columns
    assert "symptom_rarity" in transformed.columns
    assert transformed["symptom_count"].iloc[0] == 4
    assert transformed["symptom_count"].iloc[1] == 1
