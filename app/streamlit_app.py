"""
streamlit_app.py  —  Skincare Recommender (Simplified UI)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from src.data_loader import load_data, CLASS_NAMES
from src.recommend   import get_recommendation
from src.evaluate    import (radar_chart_input, confidence_bar_chart,
                             metrics_comparison_chart, confusion_matrix_plotly,
                             feature_importance_plotly)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="🌸",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource(show_spinner=False)
def load_models():
    needed = ["best_classifier.pkl","kmeans_cluster.pkl","scaler.pkl",
              "selector.pkl","label_encoder.pkl","selected_features.pkl",
              "all_results.pkl","best_model_name.pkl"]
    if any(not os.path.exists(os.path.join(MODELS_DIR, f)) for f in needed):
        return None
    return {
        "clf":       joblib.load(os.path.join(MODELS_DIR, "best_classifier.pkl")),
        "kmeans":    joblib.load(os.path.join(MODELS_DIR, "kmeans_cluster.pkl")),
        "scaler":    joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "selector":  joblib.load(os.path.join(MODELS_DIR, "selector.pkl")),
        "le":        joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl")),
        "features":  joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl")),
        "results":   joblib.load(os.path.join(MODELS_DIR, "all_results.pkl")),
        "best_name": joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl")),
    }

@st.cache_data(show_spinner=False)
def load_dataset():
    return load_data()

# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🌸 SkincareAI")
st.sidebar.write("Personalized Skincare Recommender")

page = st.sidebar.radio("Navigasi", ["Beranda", "Analisis Kulit", "Info Model"])

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Dermatology Dataset\n366 sampel · 34 fitur")

# ══════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════════════════════
if page == "Beranda":
    st.title("Sistem Rekomendasi Skincare Personal")
    st.write("Berbasis Machine Learning menggunakan UCI Dermatology Dataset.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampel Data", "366")
    c2.metric("Fitur Klinis", "34")
    c3.metric("Kondisi Kulit", "6")
    c4.metric("Model ML", "4")

    st.subheader("Bagaimana Cara Kerjanya?")
    st.write("1. **Input Gejala**: Masukkan gejala kulit Anda menggunakan formulir.")
    st.write("2. **Analisis AI**: Model Machine Learning menganalisis pola gejala.")
    st.write("3. **Rekomendasi**: Dapatkan rekomendasi produk skincare yang dipersonalisasi.")

    st.info("**Disclaimer:** Aplikasi ini dibuat untuk edukasi dan tidak menggantikan diagnosis medis profesional.")

    models = load_models()
    if models is None:
        st.warning("Model belum dilatih. Jalankan `python src/train_model.py` terlebih dahulu.")
    else:
        st.success(f"Model siap digunakan! (Best model: {models['best_name']})")

# ══════════════════════════════════════════════════════════════
# PAGE: ANALISIS KULIT & HASIL
# ══════════════════════════════════════════════════════════════
elif page == "Analisis Kulit":
    st.title("Analisis Kondisi Kulit & Rekomendasi")
    
    models = load_models()
    if models is None:
        st.error("Model belum dilatih! Jalankan `python src/train_model.py` terlebih dahulu.")
        st.stop()

    with st.form("skin_form"):
        st.subheader("Data Pribadi")
        col_a, col_b = st.columns(2)
        age          = col_a.number_input("Usia", min_value=10, max_value=80, value=22, step=1)
        family_hist  = col_b.selectbox("Riwayat keluarga dengan kondisi kulit serupa?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        st.subheader("Gejala Klinis (0: Tidak ada - 3: Parah)")
        clinical_symptoms = [
            "erythema","scaling","definite_borders","itching",
            "koebner_phenomenon","polygonal_papules","follicular_papules",
            "oral_mucosal_involvement","knee_elbow_involvement","scalp_involvement",
        ]
        labels = {
            "erythema": "Kemerahan pada kulit", "scaling": "Kulit bersisik",
            "definite_borders": "Batas lesi yang jelas", "itching": "Rasa gatal",
            "koebner_phenomenon": "Koebner phenomenon", "polygonal_papules": "Papula poligonal",
            "follicular_papules": "Papula folikular", "oral_mucosal_involvement": "Mukosa mulut",
            "knee_elbow_involvement": "Lesi di lutut/siku", "scalp_involvement": "Lesi kulit kepala",
        }
        
        vals = {}
        r1, r2 = st.columns(2)
        for i, feat in enumerate(clinical_symptoms):
            col = r1 if i % 2 == 0 else r2
            vals[feat] = col.slider(labels[feat], 0, 3, 0)

        submitted = st.form_submit_button("Analisis Sekarang")

    if submitted:
        all_features_ordered = [
            "erythema","scaling","definite_borders","itching",
            "koebner_phenomenon","polygonal_papules","follicular_papules",
            "oral_mucosal_involvement","knee_elbow_involvement","scalp_involvement",
            "family_history",
            "melanin_incontinence","eosinophils_in_infiltrate","PNL_infiltrate",
            "fibrosis_papillary_dermis","exocytosis","acanthosis","hyperkeratosis",
            "parakeratosis","clubbing_rete_ridges","elongation_rete_ridges",
            "thinning_suprapapillary_epidermis","spongiform_pustule",
            "munro_microabcess","focal_hypergranulosis",
            "disappearance_granular_layer","vacuolisation_damage_basal_layer",
            "spongiosis","saw_tooth_appearance","follicular_horn_plug",
            "perifollicular_parakeratosis","inflammatory_mononuclear_infiltrate",
            "band_like_infiltrate","age",
        ]
        raw = {f: 0.0 for f in all_features_ordered}
        for feat in clinical_symptoms:
            raw[feat] = float(vals[feat])
        raw["family_history"] = float(family_hist)
        raw["age"] = float(age)

        feat_vector = np.array([[raw[f] for f in all_features_ordered]])
        scaled      = models["scaler"].transform(feat_vector)
        selected    = models["selector"].transform(scaled)

        pred_enc   = models["clf"].predict(selected)[0]
        proba      = models["clf"].predict_proba(selected)[0]
        pred_label = models["le"].inverse_transform([pred_enc])[0]
        cond_name  = CLASS_NAMES.get(pred_label, str(pred_label))
        cluster_id = int(models["kmeans"].predict(selected)[0])
        
        # Display Results directly below the form
        st.markdown("---")
        st.header("Hasil Analisis")
        
        conf = max(proba) * 100
        rec = get_recommendation(cond_name, cluster_id, age)
        skin_prof = rec["skin_profile"]

        st.success(f"**Kondisi terdeteksi:** {cond_name} (Confidence: {conf:.1f}%)")
        st.info(f"**Tipe Kulit:** {skin_prof['name']} - {skin_prof['desc']}")
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Profil Gejala")
            short = {
                "erythema":"Kemerahan","scaling":"Bersisik",
                "definite_borders":"Batas Lesi","itching":"Gatal",
                "koebner_phenomenon":"Koebner","polygonal_papules":"Papula Polig.",
                "follicular_papules":"Papula Folik.","oral_mucosal_involvement":"Mukosa Mulut",
                "knee_elbow_involvement":"Lutut/Siku","scalp_involvement":"Kulit Kepala",
            }
            labeled = {short.get(k, k): v for k, v in vals.items()}
            st.plotly_chart(radar_chart_input(labeled), use_container_width=True)

        with col_r:
            st.subheader("Confidence per Kondisi")
            class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
            st.plotly_chart(confidence_bar_chart(class_labels, proba.tolist()), use_container_width=True)

        st.markdown("---")
        st.subheader("Rekomendasi Produk Skincare")
        for cat, prod in rec["products"].items():
            with st.expander(f"{cat.title()} - {prod['name']} (by {prod['brand']})", expanded=True):
                st.write(f"**Key Ingredients:** {', '.join(prod['ingredients'])}")
                st.write(f"**Alasan:** {prod['why']}")

        if rec["avoid"]:
            st.subheader("Bahan yang Harus Dihindari")
            for a in rec["avoid"]:
                st.warning(f"❌ {a}")

        if rec["tips"]:
            st.subheader("Tips Perawatan")
            for tip in rec["tips"]:
                st.success(f"✅ {tip}")

# ══════════════════════════════════════════════════════════════
# PAGE: INFO MODEL
# ══════════════════════════════════════════════════════════════
elif page == "Info Model":
    st.title("Informasi Model ML")
    models = load_models()
    if models is None:
        st.error("Model belum dilatih! Jalankan `python src/train_model.py`")
        st.stop()

    results   = models["results"]
    best_name = models["best_name"]
    best      = results[best_name]
    class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    st.subheader(f"Best Model: {best_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{best['accuracy']:.4f}")
    c2.metric("F1-Score", f"{best['f1']:.4f}")
    c3.metric("CV 5-Fold", f"{best['cv_mean']:.4f}")
    c4.metric("ROC-AUC", f"{best.get('auc') or 0:.4f}")

    st.subheader("Perbandingan Model")
    st.plotly_chart(metrics_comparison_chart(results), use_container_width=True)

    st.subheader("Confusion Matrix")
    sel_model = st.selectbox("Pilih Model:", list(results.keys()))
    st.plotly_chart(confusion_matrix_plotly(results[sel_model]["conf_matrix"], class_labels, sel_model), use_container_width=True)

    if "Random Forest" in results:
        st.subheader("Feature Importance (Random Forest)")
        rf = results["Random Forest"]["model"]
        st.plotly_chart(feature_importance_plotly(models["features"], rf.feature_importances_.tolist()), use_container_width=True)

    st.subheader("Tabel Metrik Lengkap")
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Accuracy":  f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall":    f"{res['recall']:.4f}",
            "F1-Score":  f"{res['f1']:.4f}",
            "CV Mean":   f"{res['cv_mean']:.4f}",
            "ROC-AUC":   f"{res['auc']:.4f}" if res["auc"] else "N/A",
            "Best":      "🏆" if name == best_name else "",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
