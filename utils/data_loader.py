"""
Data loading and caching utility functions (v2).
Includes disease-symptom mapping loader for symptom overlap analysis.
"""

import os
import pandas as pd
import streamlit as st

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data
def load_training_data():
    """Load and cache the training dataset."""
    path = os.path.join(_DATA_DIR, "Training.csv")
    if not os.path.exists(path):
        from data.prepare_dataset import main as prep_main
        prep_main()
    df = pd.read_csv(path)

    # Sanity clean up duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_data
def load_testing_data():
    """Load and cache the testing dataset."""
    path = os.path.join(_DATA_DIR, "Testing.csv")
    if not os.path.exists(path):
        from data.prepare_dataset import main as prep_main
        prep_main()
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


@st.cache_data
def load_disease_info():
    """Load disease reference metadata."""
    path = os.path.join(_DATA_DIR, "disease_info.csv")
    if not os.path.exists(path):
        from data.generate_disease_info import main as gen_info
        gen_info()
    return pd.read_csv(path)


@st.cache_data
def load_disease_symptoms_map():
    """Load disease-to-symptom mapping from prepare_dataset.py for symptom overlap analysis."""
    from data.prepare_dataset import DISEASE_SYMPTOMS
    return dict(DISEASE_SYMPTOMS)


def get_symptom_list(df=None):
    """Return list of symptom column names (excluding target)."""
    if df is None:
        df = load_training_data()
    return [c for c in df.columns if c != "prognosis"]


def get_disease_list(df=None):
    """Return sorted list of unique disease names."""
    if df is None:
        df = load_training_data()
    return sorted(df["prognosis"].unique().tolist())


def symptom_display_name(raw_name):
    """Format a symptom column name for UI display."""
    return raw_name.replace("_", " ").title()


def get_dataset_stats():
    """Return dict with current dataset statistics (disease count, symptom count)."""
    from data.prepare_dataset import DISEASE_SYMPTOMS, SYMPTOMS
    return {
        "disease_count": len(DISEASE_SYMPTOMS),
        "symptom_count": len(SYMPTOMS),
    }
