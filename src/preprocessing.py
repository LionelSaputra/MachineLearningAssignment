"""
preprocessing.py
----------------
Preprocessing pipeline: scaling, encoding, train/test split, feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the dermatology DataFrame.

    Args:
        df: Raw DataFrame from data_loader.load_data()
        test_size: Fraction of data for test split
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    drop_cols = ["diagnosis", "diagnosis_name"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(float)
    y = df["diagnosis"].values.astype(int)

    # Encode labels to 0-based index
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols, le


def select_features(X_train, X_test, y_train, feature_names, k: int = 20):
    """
    Select top-k features using ANOVA F-value.

    Returns:
        X_train_sel, X_test_sel, selector, selected_feature_names
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_names = [feature_names[i] for i in selector.get_support(indices=True)]
    return X_train_sel, X_test_sel, selector, selected_names
