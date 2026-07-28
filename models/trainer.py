"""
Model Trainer Module (OOP — Module 1)
======================================
Provides OOP-based model training for Saarthi Medical Diagnosis.
Uses only syllabus classification algorithms:
  - Logistic Regression, Decision Tree, Random Forest
  - KNN, SVM, Naive Bayes (GaussianNB)
"""

import os
import joblib
import numpy as np
import pandas as pd

# Classification Models (Module 4: ML Fundamentals)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


class ModelTrainer:
    """
    OOP class for training and managing ML models (Module 1: OOP Basics).
    """

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.trained_models = {}

    def get_models(self):
        """Returns dictionary of all 6 classification models from syllabus."""
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "decision_tree": DecisionTreeClassifier(max_depth=15, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(kernel="rbf", probability=True, random_state=42),
            "naive_bayes": GaussianNB(),
        }

    def train_models(self, X_train, y_train):
        """
        Trains all 6 models and saves them to disk.
        Uses loops and file handling (Module 1).
        """
        models = self.get_models()
        self.trained_models = {}

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Save model using joblib (Module 1: File Handling)
            filepath = os.path.join(self.models_dir, f"{name}.pkl")
            joblib.dump(model, filepath)

        return self.trained_models

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and return accuracy scores.
        Uses Module 4: Model Evaluation (Accuracy).
        """
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        return results

    def get_best_model(self, X_test, y_test):
        """Returns the name and instance of the best performing model."""
        scores = self.evaluate_models(X_test, y_test)
        best_name = max(scores, key=scores.get)
        return best_name, self.trained_models[best_name], scores[best_name]
