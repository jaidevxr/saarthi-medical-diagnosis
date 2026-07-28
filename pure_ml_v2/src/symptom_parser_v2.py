"""
Pure ML Symptom Parser (pure_ml_v2/src/symptom_parser_v2.py)
============================================================
Self-contained symptom extraction module for pure_ml_v2.
Extracts binary feature vectors from patient query text using standard string/regex matching
against the canonical 132 symptom column names, without any external project dependencies.
"""

import re
import pandas as pd

class PureMLSymptomParser:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        # Create string pattern map for each feature
        self.pattern_map = {}
        for feat in self.feature_names:
            clean_feat = feat.replace("_", " ")
            # Basic token patterns
            patterns = [clean_feat, feat]
            
            # Additional common variations for standard syllabus features
            if feat == "high_fever":
                patterns.extend(["fever", "bukhar", "high temp", "temperature"])
            elif feat == "cough":
                patterns.extend(["khansi", "coughing"])
            elif feat == "chest_pain":
                patterns.extend(["seene me dard", "chest pain", "chest tightness"])
            elif feat == "headache":
                patterns.extend(["sir me dard", "head ache", "head pain"])
            elif feat == "vomiting":
                patterns.extend(["ulti", "nausea", "vomit"])
            elif feat == "diarrhoea":
                patterns.extend(["loose motion", "watery stool", "diarrhea"])
            elif feat == "joint_pain":
                patterns.extend(["joint pain", "joints pain", "ghutno me dard"])
            elif feat == "burning_micturition":
                patterns.extend(["peshab me jalan", "burning urine", "painful urination"])
            elif feat == "yellowish_skin":
                patterns.extend(["yellow skin", "peeli skin", "jaundice"])
                
            self.pattern_map[feat] = patterns

    def parse_to_vector(self, text):
        text_lower = text.lower()
        vector = {feat: 0 for feat in self.feature_names}
        matched = []

        for feat, patterns in self.pattern_map.items():
            for pat in patterns:
                # Word boundary search
                if re.search(r'\b' + re.escape(pat) + r'\b', text_lower):
                    vector[feat] = 1
                    matched.append(feat)
                    break

        return vector, matched
