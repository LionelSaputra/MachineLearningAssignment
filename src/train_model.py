"""
train_model.py
--------------
Train, evaluate, and persist multiple classical ML classifiers + K-Means clustering.
Run this script once to generate model artifacts in /models.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_data, CLASS_NAMES
from src.preprocessing import preprocess, select_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

SKIN_PROFILE_NAMES = {
    0: "Sensitif",
    1: "Normal",
    2: "Berminyak",
    3: "Kering",
    4: "Kombinasi",
}


def get_data():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names, le = preprocess(df)
    X_train_sel, X_test_sel, selector, sel_names = select_features(
        X_train, X_test, y_train, feature_names, k=20
    )
    return (X_train, X_test, y_train, y_test, scaler, feature_names,
            X_train_sel, X_test_sel, selector, sel_names, le, df)


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test, class_labels):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        auc = None
        if y_prob is not None:
            try:
                y_bin = label_binarize(y_test, classes=list(range(len(class_labels))))
                auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                auc = None

        results[name] = {
            "model": model, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "cv_mean": cv.mean(), "cv_std": cv.std(),
            "auc": auc, "y_pred": y_pred, "y_prob": y_prob,
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }
        print(f"[{name}]  Acc={acc:.4f}  F1={f1:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")
    return results


def plot_model_comparison(results):
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.18
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(), color=color, alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison — Weighted Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(conf, class_labels, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(os.path.join(PLOTS_DIR, f"cm_{safe}.png"), dpi=150)
    plt.close()


def plot_feature_importance(rf_model, feature_names):
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [feature_names[i].replace("_", " ").title() for i in idx[::-1]],
        imp[idx[::-1]], color="#6C63FF", alpha=0.85
    )
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def plot_elbow(X_scaled):
    inertias = []
    K_range  = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(K_range), inertias, "o-", color="#6C63FF", linewidth=2, markersize=7)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for K-Means", fontsize=13, fontweight="bold")
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "elbow.png"), dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("  SKINCARE RECOMMENDER — MODEL TRAINING")
    print("=" * 60)

    (X_train, X_test, y_train, y_test, scaler, feature_names,
     X_train_sel, X_test_sel, selector, sel_names, le, df) = get_data()

    class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print(f"\nFeatures: {len(feature_names)}  |  Selected: {len(sel_names)}")
    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}\n")

    models  = build_models()
    results = evaluate_models(models, X_train_sel, X_test_sel, y_train, y_test, class_labels)

    best_name  = max(results, key=lambda n: results[n]["f1"])
    best_model = results[best_name]["model"]
    print(f"\n[BEST MODEL] {best_name}  (F1={results[best_name]['f1']:.4f})\n")

    plot_model_comparison(results)
    for name, res in results.items():
        plot_confusion_matrix(res["conf_matrix"], class_labels, name)
    if "Random Forest" in results:
        plot_feature_importance(results["Random Forest"]["model"], sel_names)

    plot_elbow(X_train_sel)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
    kmeans.fit(X_train_sel)

    joblib.dump(best_model,  os.path.join(MODELS_DIR, "best_classifier.pkl"))
    joblib.dump(kmeans,       os.path.join(MODELS_DIR, "kmeans_cluster.pkl"))
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(selector,     os.path.join(MODELS_DIR, "selector.pkl"))
    joblib.dump(le,           os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(sel_names,    os.path.join(MODELS_DIR, "selected_features.pkl"))
    joblib.dump(results,      os.path.join(MODELS_DIR, "all_results.pkl"))
    joblib.dump(best_name,    os.path.join(MODELS_DIR, "best_model_name.pkl"))
    print("[SUCCESS] All models and plots saved!")


if __name__ == "__main__":
    main()
