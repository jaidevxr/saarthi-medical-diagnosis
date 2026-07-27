"""
Encoding utilities for the target and feature columns.
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd


class DataEncoder:
    """Handles label encoding for the prognosis target column."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, y):
        """Fit the label encoder on the target column."""
        self.label_encoder.fit(y)
        self._is_fitted = True
        return self

    def transform(self, y):
        """Transform target labels to integer codes."""
        return self.label_encoder.transform(y)

    def fit_transform(self, y):
        """Fit and transform in one step."""
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, codes):
        """Convert integer codes back to disease names."""
        return self.label_encoder.inverse_transform(codes)

    def get_encoding_map(self):
        """Return a DataFrame mapping disease names to integer codes."""
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted yet. Call fit() first.")
        classes = self.label_encoder.classes_
        return pd.DataFrame({
            "Disease": classes,
            "Code": range(len(classes)),
        })

    @property
    def classes(self):
        return self.label_encoder.classes_ if self._is_fitted else []

    @property
    def n_classes(self):
        return len(self.classes)
