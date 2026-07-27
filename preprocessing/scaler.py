"""
Feature scaling utilities.
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class DataScaler:
    """Wraps StandardScaler with before/after comparison helpers."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X):
        """Fit the scaler on training features."""
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Scale features using the fitted scaler."""
        scaled = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(scaled, columns=X.columns, index=X.index)
        return scaled

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_comparison(self, X, n_cols=5):
        """
        Return a before/after comparison for the first *n_cols* features.
        Useful for demonstrating the effect of scaling.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted yet.")

        cols = X.columns[:n_cols] if isinstance(X, pd.DataFrame) else list(range(n_cols))
        X_sub = X[cols] if isinstance(X, pd.DataFrame) else X[:, :n_cols]

        before = pd.DataFrame(X_sub).describe().loc[["mean", "std", "min", "max"]]
        before.columns = cols
        before.index = ["Mean", "Std", "Min", "Max"]

        scaled = self.transform(X)
        X_s_sub = scaled[cols] if isinstance(scaled, pd.DataFrame) else scaled[:, :n_cols]
        after = pd.DataFrame(X_s_sub).describe().loc[["mean", "std", "min", "max"]]
        after.columns = cols
        after.index = ["Mean", "Std", "Min", "Max"]

        return before.round(4), after.round(4)
