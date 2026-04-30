"""
data_loader.py
--------------
Fetch and save the UCI Dermatology dataset using ucimlrepo.
Handles missing value imputation and basic cleaning.
"""

import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo


# Column names from the UCI Dermatology dataset
FEATURE_NAMES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement", "scalp_involvement",
    "family_history",
    # Histopathological
    "melanin_incontinence", "eosinophils_in_infiltrate", "PNL_infiltrate",
    "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
    "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_granular_layer", "vacuolisation_damage_basal_layer",
    "spongiosis", "saw_tooth_appearance", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate",
    "age",
]

CLASS_NAMES = {
    1: "Psoriasis",
    2: "Seboreic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris",
}

# Friendly display names for clinical symptoms (for UI)
CLINICAL_FEATURES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement", "scalp_involvement",
    "family_history", "age",
]

SYMPTOM_LABELS = {
    "erythema": "Kemerahan pada kulit (Erythema)",
    "scaling": "Kulit bersisik (Scaling)",
    "definite_borders": "Batas lesi yang jelas",
    "itching": "Rasa gatal",
    "koebner_phenomenon": "Lesi muncul di area trauma kulit",
    "polygonal_papules": "Papula berbentuk poligonal",
    "follicular_papules": "Papula folikular",
    "oral_mucosal_involvement": "Keterlibatan mukosa mulut",
    "knee_elbow_involvement": "Lesi di lutut/siku",
    "scalp_involvement": "Lesi di kulit kepala",
    "family_history": "Riwayat keluarga dengan kondisi serupa",
    "age": "Usia",
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dermatology_processed.csv")


def load_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the UCI Dermatology dataset. If already cached locally, load from CSV.
    Otherwise, fetch from ucimlrepo and save.

    Returns:
        df (pd.DataFrame): Cleaned dataset with named columns and target column 'diagnosis'.
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if os.path.exists(DATA_PATH) and not force_reload:
        df = pd.read_csv(DATA_PATH)
        return df

    # Fetch from UCI ML Repo
    dermatology = fetch_ucirepo(id=33)
    X = dermatology.data.features.copy()
    y = dermatology.data.targets.copy()

    # Rename columns
    X.columns = FEATURE_NAMES

    # Handle missing values — impute with median
    X["age"] = pd.to_numeric(X["age"], errors="coerce")
    X["age"] = X["age"].fillna(X["age"].median())
    X = X.fillna(X.median(numeric_only=True))

    # Combine
    df = X.copy()
    df["diagnosis"] = y.values.flatten().astype(int)
    df["diagnosis_name"] = df["diagnosis"].map(CLASS_NAMES)

    # Save locally
    df.to_csv(DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_data(force_reload=True)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df["diagnosis_name"].value_counts())
