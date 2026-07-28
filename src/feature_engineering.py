"""
Feature Engineering Module (src/feature_engineering.py)
======================================================
Engineers classical domain-informed features:
1. Symptom count
2. Body system count
3. Symptom rarity score
4. Common vs uncommon symptom ratio
5. Manual severity score
6. Symptom overlap score per disease
"""

import os
import json
import numpy as np
import pandas as pd

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

class FeatureExtractor:
    def __init__(self,
                 body_systems_file=os.path.join(CONFIG_DIR, "body_systems.json"),
                 severity_file=os.path.join(CONFIG_DIR, "symptom_severity.json")):
        
        self.body_systems = {}
        if os.path.exists(body_systems_file):
            with open(body_systems_file, "r", encoding="utf-8") as f:
                self.body_systems = json.load(f)
                
        self.severity_map = {}
        if os.path.exists(severity_file):
            with open(severity_file, "r", encoding="utf-8") as f:
                self.severity_map = json.load(f)
                
        self.symptom_frequencies = {}
        self.median_freq = 0.0
        self.disease_typical_symptoms = {}

    def fit(self, X_train_df, y_train_series=None):
        """
        Fit frequency statistics and typical disease symptom mappings from training data.
        X_train_df: DataFrame of binary symptom columns
        """
        symptom_cols = list(X_train_df.columns)
        
        # Calculate symptom rarity/frequencies
        counts = X_train_df.sum(axis=0)
        total_rows = len(X_train_df)
        
        for col in symptom_cols:
            freq = (counts[col] + 1) / (total_rows + 1)
            self.symptom_frequencies[col] = freq
            
        self.median_freq = np.median(list(self.symptom_frequencies.values()))
        
        # Compute typical symptoms per disease if y is provided
        if y_train_series is not None:
            df = X_train_df.copy()
            df["prognosis"] = y_train_series.values
            for disease, group in df.groupby("prognosis"):
                active_symptoms = group[symptom_cols].mean(axis=0)
                typical = list(active_symptoms[active_symptoms > 0.1].index)
                self.disease_typical_symptoms[disease] = typical

    def transform(self, X_df, active_feature_set=None):
        """
        Extracts engineered features for X_df (binary DataFrame).
        active_feature_set: list of feature names to include. Options:
          ['symptom_count', 'body_system_count', 'symptom_rarity', 'common_uncommon_ratio', 'severity_score', 'overlap_score']
        """
        if active_feature_set is None:
            active_feature_set = ['symptom_count', 'body_system_count', 'symptom_rarity', 'common_uncommon_ratio', 'severity_score', 'overlap_score']

        symptom_cols = [c for c in X_df.columns if c in self.symptom_frequencies]
        X_mat = X_df[symptom_cols].values
        n_samples = len(X_df)
        
        extra_features = []
        feature_names = []

        # 1. Symptom count
        if 'symptom_count' in active_feature_set:
            counts = X_mat.sum(axis=1, keepdims=True)
            extra_features.append(counts)
            feature_names.append('symptom_count')

        # 2. Body system count
        if 'body_system_count' in active_feature_set:
            sys_counts = []
            for i in range(n_samples):
                active = [symptom_cols[j] for j in np.where(X_mat[i] == 1)[0]]
                systems = set()
                for s in active:
                    for sys_name, sys_syms in self.body_systems.items():
                        if s in sys_syms:
                            systems.add(sys_name)
                sys_counts.append(len(systems))
            extra_features.append(np.array(sys_counts).reshape(-1, 1))
            feature_names.append('body_system_count')

        # 3. Symptom rarity score
        if 'symptom_rarity' in active_feature_set:
            rarities = []
            for i in range(n_samples):
                active = [symptom_cols[j] for j in np.where(X_mat[i] == 1)[0]]
                rarity = sum(1.0 / self.symptom_frequencies.get(s, 0.05) for s in active)
                rarities.append(rarity)
            extra_features.append(np.array(rarities).reshape(-1, 1))
            feature_names.append('symptom_rarity')

        # 4. Common vs uncommon ratio
        if 'common_uncommon_ratio' in active_feature_set:
            ratios = []
            for i in range(n_samples):
                active = [symptom_cols[j] for j in np.where(X_mat[i] == 1)[0]]
                if not active:
                    ratios.append(0.0)
                    continue
                common = sum(1 for s in active if self.symptom_frequencies.get(s, 0) > self.median_freq)
                uncommon = len(active) - common
                ratio = (common + 1.0) / (uncommon + 1.0)
                ratios.append(ratio)
            extra_features.append(np.array(ratios).reshape(-1, 1))
            feature_names.append('common_uncommon_ratio')

        # 5. Manual severity score
        if 'severity_score' in active_feature_set:
            severities = []
            for i in range(n_samples):
                active = [symptom_cols[j] for j in np.where(X_mat[i] == 1)[0]]
                sev = sum(self.severity_map.get(s, self.severity_map.get("default", 1)) for s in active)
                severities.append(sev)
            extra_features.append(np.array(severities).reshape(-1, 1))
            feature_names.append('severity_score')

        # 6. Symptom overlap score (max overlap fraction across disease templates)
        if 'overlap_score' in active_feature_set:
            overlaps = []
            for i in range(n_samples):
                active = set([symptom_cols[j] for j in np.where(X_mat[i] == 1)[0]])
                if not active or not self.disease_typical_symptoms:
                    overlaps.append(0.0)
                    continue
                max_ov = 0.0
                for disease, typical_set in self.disease_typical_symptoms.items():
                    if typical_set:
                        ov = len(active.intersection(typical_set)) / len(typical_set)
                        if ov > max_ov:
                            max_ov = ov
                overlaps.append(max_ov)
            extra_features.append(np.array(overlaps).reshape(-1, 1))
            feature_names.append('overlap_score')

        if extra_features:
            extra_mat = np.hstack(extra_features)
            X_combined = np.hstack([X_mat, extra_mat])
            all_names = list(symptom_cols) + feature_names
            return pd.DataFrame(X_combined, columns=all_names)
        else:
            return X_df[symptom_cols].copy()
