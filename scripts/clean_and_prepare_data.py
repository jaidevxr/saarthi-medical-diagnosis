"""
Data Cleaning, Deduplication, Cross-Class Conflict Resolution, and Leakage-Free Splitting
========================================================================================
1. Deduplicates exact duplicate rows.
2. Resolves cross-class duplicates (same symptom binary vector mapped to multiple diseases).
3. Re-splits cleanly into Training.csv (80%) and Testing.csv (20%) using Stratified Split.
4. Verifies 0% train-test vector leakage.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def clean_and_split_data():
    print("=" * 70)
    print("STEP 1: DATA CLEANING, DEDUPLICATION & LEAKAGE-FREE SPLITTING")
    print("=" * 70)

    # 1. Load data
    train_path = os.path.join(DATA_DIR, "Training.csv")
    test_path = os.path.join(DATA_DIR, "Testing.csv")
    
    if not os.path.exists(train_path):
        from data.prepare_dataset import main as prep_main
        prep_main()

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Deduplicate columns if any
    df_train = df_train.loc[:, ~df_train.columns.duplicated()]
    df_test = df_test.loc[:, ~df_test.columns.duplicated()]

    full_df = pd.concat([df_train, df_test], ignore_index=True)
    initial_count = len(full_df)
    print(f"  Initial raw row count (Train + Test): {initial_count}")

    symptom_cols = [c for c in full_df.columns if c != "prognosis"]

    # 2. Drop exact duplicates (feature vector + label)
    df_dedup = full_df.drop_duplicates(keep="first").reset_index(drop=True)
    dedup_count = len(df_dedup)
    dropped_exact = initial_count - dedup_count
    print(f"  Exact duplicates dropped: {dropped_exact} ({dropped_exact / initial_count * 100:.1f}%)")
    print(f"  Remaining rows after exact dedup: {dedup_count}")

    # 3. Resolve cross-class conflicts (same symptom vector, different prognosis)
    cross_class = df_dedup.groupby(symptom_cols)["prognosis"].nunique()
    conflict_vectors = cross_class[cross_class > 1].index

    if len(conflict_vectors) > 0:
        print(f"  Cross-class conflict vectors found: {len(conflict_vectors)}")
        # Drop rows corresponding to conflicting symptom vectors to remove label noise
        conflict_mask = df_dedup.set_index(symptom_cols).index.isin(conflict_vectors)
        df_clean = df_dedup[~conflict_mask].reset_index(drop=True)
        dropped_conflicts = len(df_dedup) - len(df_clean)
        print(f"  Dropped conflict rows: {dropped_conflicts}")
    else:
        df_clean = df_dedup
        print("  Cross-class conflict vectors found: 0")

    final_clean_count = len(df_clean)
    print(f"  Final clean unique dataset size: {final_clean_count}")
    print(f"  Diseases represented: {df_clean['prognosis'].nunique()}")

    # 4. Perform Stratified Train/Test Split (80/20) with seed 42
    X = df_clean[symptom_cols]
    y = df_clean["prognosis"]

    # Check minimum class frequency
    class_counts = y.value_counts()
    min_count = class_counts.min()
    print(f"  Min samples per disease class: {min_count}, Max: {class_counts.max()}")

    if min_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

    # 5. Verify Zero Leakage (exact feature vector overlap between train and test)
    merged_leak = pd.merge(X_train, X_test, on=symptom_cols, how="inner")
    leak_count = len(merged_leak)
    print(f"  Train-Test Exact Vector Leakage Check: {leak_count} rows ({leak_count / len(X_test) * 100:.2f}%)")
    assert leak_count == 0, f"Error: Leakage detected ({leak_count} rows)!"

    # 6. Save clean splits
    train_clean_df = pd.concat([X_train, y_train], axis=1)
    test_clean_df = pd.concat([X_test, y_test], axis=1)

    train_clean_df.to_csv(os.path.join(DATA_DIR, "Training.csv"), index=False)
    test_clean_df.to_csv(os.path.join(DATA_DIR, "Testing.csv"), index=False)

    print(f"  [OK] Saved clean data/Training.csv: {len(train_clean_df)} rows")
    print(f"  [OK] Saved clean data/Testing.csv:  {len(test_clean_df)} rows")
    print("=" * 70)

if __name__ == "__main__":
    clean_and_split_data()
