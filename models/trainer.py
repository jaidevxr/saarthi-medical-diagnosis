"""
Model Trainer Module
====================
Provides model training abstractions for Saarthi.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


class ModelTrainer:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir

    def train_models(self, X_train, y_train):
        """
        Trains baseline and calibrated ensemble models.
        """
        models = {
            "calibrated_nb": CalibratedClassifierCV(MultinomialNB(alpha=0.1), cv=5),
            "random_forest_tuned": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
        }

        trained = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained[name] = model
            joblib.dump(model, os.path.join(self.models_dir, f"{name}.pkl"))

        return trained
