"""
Data cleaning utilities.
Checks for missing values, duplicates, and outliers.
Provides before/after comparison summaries.
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """Stateless helper that analyses and cleans a DataFrame."""

    def __init__(self, df):
        self.original = df.copy()
        self.cleaned = df.copy()
        self.log = []  # ordered list of cleaning steps

    # ── Missing Values ──────────────────────────────
    def check_missing(self):
        """Return a summary DataFrame of missing values per column."""
        missing = self.original.isnull().sum()
        pct = (missing / len(self.original) * 100).round(2)
        summary = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": pct.values,
        })
        summary = summary[summary["Missing Count"] > 0].sort_values(
            "Missing Count", ascending=False
        )
        return summary

    def handle_missing(self):
        """Drop rows with any missing values (dataset is binary so imputation is not appropriate)."""
        before = len(self.cleaned)
        self.cleaned = self.cleaned.dropna()
        after = len(self.cleaned)
        removed = before - after
        self.log.append(f"Missing values: removed {removed} rows ({before} -> {after})")
        return removed

    # ── Duplicates ──────────────────────────────────
    def check_duplicates(self):
        """Return the number of duplicate rows."""
        return self.original.duplicated().sum()

    def remove_duplicates(self):
        """Remove duplicate rows."""
        before = len(self.cleaned)
        self.cleaned = self.cleaned.drop_duplicates()
        after = len(self.cleaned)
        removed = before - after
        self.log.append(f"Duplicates: removed {removed} rows ({before} -> {after})")
        return removed

    # ── Outliers ────────────────────────────────────
    def detect_outliers_iqr(self, columns=None):
        """
        Detect outliers using the IQR method on numeric columns.
        For binary (0/1) data the IQR is typically 0-1 so outliers
        are values outside {0, 1}.
        Returns a dict  column -> count of outlier rows.
        """
        if columns is None:
            columns = self.cleaned.select_dtypes(include=[np.number]).columns.tolist()

        outlier_counts = {}
        for col in columns:
            q1 = self.cleaned[col].quantile(0.25)
            q3 = self.cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (self.cleaned[col] < lower) | (self.cleaned[col] > upper)
            count = mask.sum()
            if count > 0:
                outlier_counts[col] = int(count)
        return outlier_counts

    # ── Summary ─────────────────────────────────────
    def get_cleaning_summary(self):
        """Return a dict comparing the original and cleaned DataFrames."""
        return {
            "original_rows": len(self.original),
            "cleaned_rows": len(self.cleaned),
            "original_cols": len(self.original.columns),
            "cleaned_cols": len(self.cleaned.columns),
            "original_missing": int(self.original.isnull().sum().sum()),
            "cleaned_missing": int(self.cleaned.isnull().sum().sum()),
            "original_duplicates": int(self.original.duplicated().sum()),
            "cleaned_duplicates": int(self.cleaned.duplicated().sum()),
            "steps": self.log,
        }

    def get_cleaned_data(self):
        """Return the cleaned DataFrame."""
        return self.cleaned.copy()
