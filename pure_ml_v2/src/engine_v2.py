"""
Pure ML Diagnosis Engine (pure_ml_v2/src/engine_v2.py)
=====================================================
Self-contained diagnosis engine for pure_ml_v2.
Uses trained scikit-learn model for predict_proba() and sklearn.tree.DecisionTreeClassifier
for dynamic clarifying question selection, without any hand-written entropy formulas or external dependencies.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PURE_ML_DIR = os.path.join(PROJECT_ROOT, "pure_ml_v2")
DATA_DIR = os.path.join(PURE_ML_DIR, "data")
MODELS_DIR = os.path.join(PURE_ML_DIR, "models")

from pure_ml_v2.src.symptom_parser_v2 import PureMLSymptomParser


def format_generic_question(symptom_name):
    clean_name = symptom_name.replace("_", " ")
    return f"Are you experiencing {clean_name}?"


class PureMLDiagnosisEngine:
    def __init__(self):
        # Load best trained ML model
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        payload = joblib.load(model_path)

        self.model_name = payload["model_name"]
        self.model = payload["model_object"]
        self.feature_names = payload["feature_names"]
        self.classes = list(payload["classes"])

        # Load sparse training dataset to build disease profiles for candidate DT fitting
        sparse_path = os.path.join(DATA_DIR, "training_data_sparse.csv")
        self.sparse_df = pd.read_csv(sparse_path)

        # Build disease profiles
        self.disease_profiles = {}
        for d in self.classes:
            sub = self.sparse_df[self.sparse_df["prognosis"] == d]
            self.disease_profiles[d] = sub[self.feature_names]

        # Parser
        self.parser = PureMLSymptomParser(self.feature_names)

    def select_question_with_decision_tree(self, candidate_diseases, asked_symptoms):
        """
        Uses an actual sklearn DecisionTreeClassifier fit on candidate subset rows
        to determine the optimal root feature split (tree_.feature[0]).
        """
        if len(candidate_diseases) <= 1:
            return None

        # Filter candidate rows from sparse training set
        sub_df = self.sparse_df[self.sparse_df["prognosis"].isin(candidate_diseases)]
        if len(sub_df) == 0:
            return None

        # Unasked feature columns
        available_features = [f for f in self.feature_names if f not in asked_symptoms]
        if not available_features:
            return None

        X_sub = sub_df[available_features]
        y_sub = sub_df["prognosis"]

        # Check if there is any feature variance across candidate subset
        if X_sub.nunique().max() <= 1:
            return None

        # Train a real sklearn DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        dt.fit(X_sub, y_sub)

        # Read root split feature directly from tree structure
        root_feature_idx = dt.tree_.feature[0]
        if root_feature_idx < 0:
            return None

        best_symptom = available_features[root_feature_idx]
        return best_symptom

    def predict_initial(self, text, margin_threshold=0.05):
        vector_dict, matched = self.parser.parse_to_vector(text)
        X_vec = pd.DataFrame([vector_dict])[self.feature_names]

        # ML Probability Prediction
        probas = self.model.predict_proba(X_vec)[0]
        class_probas = dict(zip(self.classes, probas))
        sorted_cand = sorted(class_probas.items(), key=lambda x: x[1], reverse=True)

        top1_p = sorted_cand[0][1]
        top2_p = sorted_cand[1][1] if len(sorted_cand) > 1 else 0.0
        margin = top1_p - top2_p

        top1 = sorted_cand[0][0]
        top3 = [pair[0] for pair in sorted_cand[:3]]

        # Candidate selection based on margin threshold
        candidates = [pair[0] for pair in sorted_cand if (top1_p - pair[1]) <= margin_threshold or pair[1] >= (top1_p - margin_threshold)]
        
        # If margin is small or candidates > 1, ask follow-up question
        asked_symptoms = set(matched)
        next_question_sym = None
        if margin < margin_threshold or len(candidates) > 1:
            next_question_sym = self.select_question_with_decision_tree(candidates, asked_symptoms)

        is_resolved = len(candidates) <= 1 or next_question_sym is None

        return {
            "initial_text": text,
            "matched_symptoms": matched,
            "vector_dict": vector_dict,
            "present_symptoms": list(matched),
            "absent_symptoms": [],
            "asked_symptoms": list(asked_symptoms),
            "rounds_asked": 0,
            "candidate_diseases": candidates,
            "candidates": candidates,
            "primary_diagnosis": top1,
            "top3_differential": top3,
            "max_probability": float(top1_p),
            "top2_probability": float(top2_p),
            "probability_margin": float(margin),
            "margin_threshold": float(margin_threshold),
            "next_question_symptom": next_question_sym,
            "next_question_text": format_generic_question(next_question_sym) if next_question_sym else None,
            "is_resolved": is_resolved
        }

    def process_followup(self, present_symptoms=None, absent_symptoms=None, asked_symptoms=None, user_answer_yes_no="NO", current_question_symptom=None, rounds_asked=0, vector_dict=None, margin_threshold=0.05):
        if vector_dict is not None:
            vector_dict = dict(vector_dict)
        else:
            vector_dict = {f: 0 for f in self.feature_names}
            for s in (present_symptoms or []):
                if s in vector_dict:
                    vector_dict[s] = 1

        asked_set = set(asked_symptoms or [])
        present_set = set(present_symptoms or [])
        absent_set = set(absent_symptoms or [])

        if current_question_symptom:
            asked_set.add(current_question_symptom)
            if user_answer_yes_no.upper() in ["YES", "Y", "TRUE", "1"]:
                vector_dict[current_question_symptom] = 1
                present_set.add(current_question_symptom)
            else:
                vector_dict[current_question_symptom] = 0
                absent_set.add(current_question_symptom)

        X_vec = pd.DataFrame([vector_dict])[self.feature_names]
        probas = self.model.predict_proba(X_vec)[0]
        class_probas = dict(zip(self.classes, probas))
        sorted_cand = sorted(class_probas.items(), key=lambda x: x[1], reverse=True)

        top1_p = sorted_cand[0][1]
        top2_p = sorted_cand[1][1] if len(sorted_cand) > 1 else 0.0
        margin = top1_p - top2_p

        top1 = sorted_cand[0][0]
        top3 = [pair[0] for pair in sorted_cand[:3]]

        candidates = [pair[0] for pair in sorted_cand if (top1_p - pair[1]) <= margin_threshold or pair[1] >= (top1_p - margin_threshold)]

        rounds_asked += 1
        next_question_sym = None
        if (margin < margin_threshold or len(candidates) > 1) and rounds_asked < 2:
            next_question_sym = self.select_question_with_decision_tree(candidates, asked_set)

        is_resolved = len(candidates) <= 1 or rounds_asked >= 2 or next_question_sym is None

        matched_list = [f for f, v in vector_dict.items() if v == 1]

        return {
            "matched_symptoms": matched_list,
            "vector_dict": vector_dict,
            "present_symptoms": list(present_set),
            "absent_symptoms": list(absent_set),
            "asked_symptoms": list(asked_set),
            "rounds_asked": rounds_asked,
            "candidate_diseases": candidates,
            "candidates": candidates,
            "primary_diagnosis": top1,
            "top3_differential": top3,
            "max_probability": float(top1_p),
            "top2_probability": float(top2_p),
            "probability_margin": float(margin),
            "margin_threshold": float(margin_threshold),
            "next_question_symptom": next_question_sym,
            "next_question_text": format_generic_question(next_question_sym) if next_question_sym else None,
            "is_resolved": is_resolved
        }
