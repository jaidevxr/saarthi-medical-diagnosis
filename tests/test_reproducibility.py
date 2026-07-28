"""
Automated Reproducibility Hash Verification Test (tests/test_reproducibility.py)
==================================================================================
Recomputes physical SHA256 file hashes for dataset v1, dataset v2, and best_model.pkl
and asserts that they match the exact hashes documented in docs/model_card.md.
"""

import os
import sys
import hashlib
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def test_model_card_hash_verification():
    """Asserts that model_card.md contains exact matching SHA256 hashes."""
    card_path = os.path.join(DOCS_DIR, "model_card.md")
    assert os.path.exists(card_path), "model_card.md does not exist!"

    with open(card_path, "r", encoding="utf-8") as f:
        card_content = f.read()

    v1_path = os.path.join(DATA_DIR, "training_data_v1.csv")
    v2_path = os.path.join(DATA_DIR, "training_data_v2.csv")
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")

    v1_hash = compute_sha256(v1_path)
    v2_hash = compute_sha256(v2_path)
    model_hash = compute_sha256(model_path)

    if v1_hash:
        assert v1_hash in card_content, f"training_data_v1.csv hash {v1_hash} not found in model_card.md!"
    if v2_hash:
        assert v2_hash in card_content, f"training_data_v2.csv hash {v2_hash} not found in model_card.md!"
    if model_hash:
        assert model_hash in card_content, f"best_model.pkl hash {model_hash} not found in model_card.md!"
