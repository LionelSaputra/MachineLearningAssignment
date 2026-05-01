"""
streamlit_app.py  —  Skincare Recommender (Part 1: imports, config, CSS)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from src.data_loader import load_data, CLASS_NAMES, SYMPTOM_LABELS, CLINICAL_FEATURES
from src.recommend   import get_recommendation, SKIN_PROFILES
from src.evaluate    import (radar_chart_input, confidence_bar_chart,
                             metrics_comparison_chart, confusion_matrix_plotly,
                             feature_importance_plotly)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkincareAI — Rekomendasi Skincare Personal",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: linear-gradient(135deg,#0F0F1A 0%,#12122A 50%,#0D1117 100%); color:#E0E0FF; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1A1A2E,#16213E) !important;
    border-right: 1px solid rgba(108,99,255,.25);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(108,99,255,.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: box-shadow .3s;
}
.card:hover { box-shadow: 0 0 24px rgba(108,99,255,.25); }

.product-card {
    background: rgba(108,99,255,.08);
    border: 1px solid rgba(108,99,255,.3);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: .8rem;
}

.metric-card {
    background: rgba(255,255,255,.04);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,.08);
}
.metric-value { font-size:2rem; font-weight:700; color:#6C63FF; }
.metric-label { font-size:.8rem; color:#9090B0; margin-top:.2rem; }

/* ── Hero ── */
.hero-title {
    font-family:'Outfit',sans-serif;
    font-size:3.2rem; font-weight:700;
    background:linear-gradient(90deg,#6C63FF,#FF6584,#43D9AD);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.2; margin-bottom:.5rem;
}
.hero-sub { font-size:1.1rem; color:#9090C0; margin-bottom:2rem; }

/* ── Badges / Pills ── */
.badge {
    display:inline-block; padding:.3rem .9rem;
    border-radius:99px; font-size:.78rem; font-weight:600;
    margin:.2rem; cursor:default;
}
.badge-purple { background:rgba(108,99,255,.2); color:#A9A4FF; border:1px solid #6C63FF; }
.badge-pink   { background:rgba(255,101,132,.2); color:#FF9DAE; border:1px solid #FF6584; }
.badge-green  { background:rgba(67,217,173,.2);  color:#7FF5D8; border:1px solid #43D9AD; }
.badge-orange { background:rgba(255,179,71,.2);  color:#FFD080; border:1px solid #FFB347; }

/* ── Section headings ── */
.section-title {
    font-family:'Outfit',sans-serif; font-size:1.5rem; font-weight:700;
    color:#E0E0FF; margin:1.5rem 0 .8rem; letter-spacing:.5px;
}

/* ── Divider ── */
.divider { border:none; border-top:1px solid rgba(255,255,255,.08); margin:1.5rem 0; }

/* ── Tip box ── */
.tip-box {
    background:rgba(67,217,173,.08); border-left:3px solid #43D9AD;
    border-radius:0 10px 10px 0; padding:.8rem 1rem; margin:.5rem 0;
    font-size:.9rem; color:#C0FFF0;
}
.warn-box {
    background:rgba(255,101,132,.08); border-left:3px solid #FF6584;
    border-radius:0 10px 10px 0; padding:.8rem 1rem; margin:.5rem 0;
    font-size:.9rem; color:#FFD0D8;
}

/* ── Slider labels ── */
.stSlider > label { color:#C0C0E0 !important; font-size:.9rem !important; }

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(90deg,#6C63FF,#9C6ADE);
    color:white; border:none; border-radius:10px;
    padding:.6rem 2rem; font-weight:600; font-size:1rem;
    transition:all .3s;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(108,99,255,.4);
}
</style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem;'>
      <span style='font-size:2.5rem;'>🌸</span>
      <div style='font-family:Outfit;font-size:1.3rem;font-weight:700;
           background:linear-gradient(90deg,#6C63FF,#FF6584);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        SkincareAI
      </div>
      <div style='font-size:.75rem;color:#6060A0;margin-top:.2rem;'>
        Personalized Skincare Recommender
      </div>
    </div>
    <hr style='border-color:rgba(108,99,255,.2);'>
    """, unsafe_allow_html=True)

    page = st.radio("Nav", ["🏠  Beranda","🔬  Analisis Kulit",
                             "💊  Hasil & Rekomendasi","📊  Info Model"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(108,99,255,.2);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.72rem;color:#606080;line-height:1.7;padding:.4rem 0;'>
      <b style='color:#8080B0;'>Dataset:</b><br>
      UCI Dermatology Dataset<br>366 sampel · 34 fitur<br>
      <a href='https://archive.ics.uci.edu/dataset/33/dermatology'
         style='color:#6C63FF;' target='_blank'>🔗 Sumber UCI</a>
    </div>""", unsafe_allow_html=True)

PAGE = page.split("  ")[1].strip()

# ══════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════════════════════
if PAGE == "Beranda":
    st.markdown("""
    <div class='hero-title'>🌸 SkincareAI</div>
    <div class='hero-sub'>Sistem Rekomendasi Skincare Personal untuk Anak Muda<br>
    Berbasis Machine Learning · UCI Dermatology Dataset</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "366",  "Sampel Data"),
        (c2, "34",   "Fitur Klinis"),
        (c3, "6",    "Kondisi Kulit"),
        (c4, "4",    "Model ML"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🤔 Bagaimana Cara Kerjanya?</div>", unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)
    for col, icon, title, desc in [
        (ca, "1️⃣", "Input Gejala", "Masukkan gejala kulit Anda menggunakan slider intuitif berdasarkan intensitas 0–3."),
        (cb, "2️⃣", "Analisis AI",  "Model Machine Learning menganalisis pola gejala dan mengklasifikasi kondisi kulit Anda."),
        (cc, "3️⃣", "Rekomendasi", "Dapatkan rekomendasi produk skincare yang dipersonalisasi sesuai kondisi & tipe kulit Anda."),
    ]:
        col.markdown(f"""
        <div class='card'>
          <div style='font-size:2rem;'>{icon}</div>
          <div style='font-weight:600;font-size:1rem;margin:.5rem 0;color:#C0C0FF;'>{title}</div>
          <div style='font-size:.88rem;color:#8080A0;line-height:1.6;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🧬 Kondisi Kulit yang Dapat Dideteksi</div>", unsafe_allow_html=True)
    conditions = {
        "Psoriasis": ("badge-purple", "Peradangan kronis dengan sisik tebal"),
        "Seboreic Dermatitis": ("badge-orange", "Kulit berminyak & berketombe"),
        "Lichen Planus": ("badge-pink", "Lesi berpola dan peradangan"),
        "Pityriasis Rosea": ("badge-green", "Ruam merah sementara"),
        "Chronic Dermatitis": ("badge-purple", "Eksim & iritasi berkepanjangan"),
        "Pityriasis Rubra Pilaris": ("badge-pink", "Gangguan keratinisasi langka"),
    }
    badge_html = "".join(
        f"<span class='badge {cls}' title='{desc}'>{name}</span>"
        for name, (cls, desc) in conditions.items()
    )
    st.markdown(f"<div style='margin:.5rem 0;'>{badge_html}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
      <div style='font-weight:600;color:#C0C0FF;margin-bottom:.5rem;'>⚠️ Disclaimer</div>
      <div style='font-size:.85rem;color:#8080A0;line-height:1.7;'>
        Aplikasi ini dibuat untuk tujuan <b>edukasi dan penelitian akademis</b>. 
        Rekomendasi yang diberikan <b>tidak menggantikan konsultasi medis profesional</b>. 
        Untuk diagnosis dan perawatan kulit yang tepat, selalu konsultasikan dengan dokter kulit (dermatologis).
      </div>
    </div>""", unsafe_allow_html=True)

    models = load_models()
    if models is None:
        st.warning("⚙️ Model belum dilatih. Pergi ke terminal dan jalankan: `python src/train_model.py`")
    else:
        st.success(f"✅ Model siap digunakan! Best model: **{models['best_name']}**")


# ══════════════════════════════════════════════════════════════
# PAGE: ANALISIS KULIT
# ══════════════════════════════════════════════════════════════
elif PAGE == "Analisis Kulit":
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>🔬 Analisis Kondisi Kulit</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Isi formulir berikut berdasarkan kondisi kulit Anda saat ini.</div>", unsafe_allow_html=True)

    models = load_models()
    if models is None:
        st.error("❌ Model belum dilatih! Jalankan dulu: `python src/train_model.py` di terminal.")
        st.stop()

    with st.form("skin_form"):
        st.markdown("<div class='section-title'>📋 Data Pribadi</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        age          = col_a.number_input("Usia", min_value=10, max_value=80, value=22, step=1)
        family_hist  = col_b.selectbox("Riwayat keluarga dengan kondisi kulit serupa?", [0, 1],
                                        format_func=lambda x: "Ya" if x else "Tidak")

        st.markdown("<div class='section-title'>🩺 Gejala Klinis</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.85rem;color:#8080A0;margin-bottom:.8rem;'>"
                    "Nilai: 0 = Tidak ada · 1 = Ringan · 2 = Sedang · 3 = Parah</div>",
                    unsafe_allow_html=True)

        clinical_symptoms = [
            "erythema","scaling","definite_borders","itching",
            "koebner_phenomenon","polygonal_papules","follicular_papules",
            "oral_mucosal_involvement","knee_elbow_involvement","scalp_involvement",
        ]
        labels = {
            "erythema":                   "Kemerahan pada kulit",
            "scaling":                    "Kulit bersisik",
            "definite_borders":           "Batas lesi yang jelas",
            "itching":                    "Rasa gatal",
            "koebner_phenomenon":         "Lesi muncul di area trauma kulit",
            "polygonal_papules":          "Papula berbentuk poligonal",
            "follicular_papules":         "Papula folikular",
            "oral_mucosal_involvement":   "Keterlibatan mukosa mulut",
            "knee_elbow_involvement":     "Lesi di lutut / siku",
            "scalp_involvement":          "Lesi di kulit kepala",
        }
        vals = {}
        r1, r2 = st.columns(2)
        for i, feat in enumerate(clinical_symptoms):
            col = r1 if i % 2 == 0 else r2
            vals[feat] = col.slider(labels[feat], 0, 3, 0, key=f"sl_{feat}")

        submitted = st.form_submit_button("🚀 Analisis Sekarang", use_container_width=True)

    if submitted:
        # Build full feature vector (34 features)
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

        st.session_state["result"] = {
            "cond_name":  cond_name,
            "cluster_id": cluster_id,
            "proba":      proba.tolist(),
            "age":        age,
            "symptoms":   vals,
        }
        st.success("✅ Analisis selesai! Buka halaman **💊 Hasil & Rekomendasi** di sidebar.")
        st.balloons()


# ══════════════════════════════════════════════════════════════
# PAGE: HASIL & REKOMENDASI
# ══════════════════════════════════════════════════════════════
elif PAGE == "Hasil & Rekomendasi":
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>💊 Hasil & Rekomendasi</div>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.info("ℹ️ Belum ada hasil analisis. Silakan buka halaman **🔬 Analisis Kulit** terlebih dahulu.")
        st.stop()

    r          = st.session_state["result"]
    cond_name  = r["cond_name"]
    cluster_id = r["cluster_id"]
    proba      = r["proba"]
    age        = r["age"]
    symptoms   = r["symptoms"]
    rec        = get_recommendation(cond_name, cluster_id, age)
    skin_prof  = rec["skin_profile"]

    conf = max(proba) * 100
    st.markdown(f"""
    <div class='card' style='border-color:#6C63FF;background:rgba(108,99,255,.08);'>
      <div style='font-size:2.2rem;font-weight:700;color:#A9A4FF;'>{cond_name}</div>
      <div style='color:#8080A0;margin:.4rem 0;'>Confidence: <b style='color:#43D9AD;'>{conf:.1f}%</b></div>
      <div>
        <span class='badge badge-green'>{skin_prof['icon']} {skin_prof['name']}</span>
        <span class='badge badge-purple'>Usia {age} thn</span>
        <span class='badge badge-pink'>{rec['concern']}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-title'>🕸️ Profil Gejala</div>", unsafe_allow_html=True)
        short = {
            "erythema":"Kemerahan","scaling":"Bersisik",
            "definite_borders":"Batas Lesi","itching":"Gatal",
            "koebner_phenomenon":"Koebner","polygonal_papules":"Papula Polig.",
            "follicular_papules":"Papula Folik.","oral_mucosal_involvement":"Mukosa Mulut",
            "knee_elbow_involvement":"Lutut/Siku","scalp_involvement":"Kulit Kepala",
        }
        labeled = {short.get(k, k): v for k, v in symptoms.items()}
        st.plotly_chart(radar_chart_input(labeled), use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title'>📊 Confidence per Kondisi</div>", unsafe_allow_html=True)
        class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
        st.plotly_chart(confidence_bar_chart(class_labels, proba), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card'>
      <span style='font-size:2.2rem;'>{skin_prof['icon']}</span>
      <span style='font-size:1.2rem;font-weight:700;color:#A9A4FF;margin-left:.5rem;'>{skin_prof['name']}</span>
      <div style='color:#8080A0;margin-top:.4rem;font-size:.9rem;'>{skin_prof['desc']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🛍️ Rekomendasi Produk Skincare</div>", unsafe_allow_html=True)
    cat_icons = {"cleanser":"🧼","toner":"💧","moisturizer":"🧴","sunscreen":"☀️","serum":"✨"}
    cat_names = {"cleanser":"Cleanser","toner":"Toner / Essence","moisturizer":"Moisturizer",
                 "sunscreen":"Sunscreen","serum":"Serum / Treatment"}

    for cat, prod in rec["products"].items():
        ingr_str = " · ".join(prod["ingredients"])
        st.markdown(f"""
        <div class='product-card'>
          <div style='font-size:.75rem;font-weight:600;color:#6C63FF;text-transform:uppercase;letter-spacing:1px;'>
            {cat_icons.get(cat,'🌿')} {cat_names.get(cat,cat.title())}
          </div>
          <div style='font-size:1rem;font-weight:700;color:#E0E0FF;margin:.3rem 0;'>{prod['name']}</div>
          <div style='font-size:.8rem;color:#9090B0;margin-bottom:.4rem;'>by {prod['brand']}</div>
          <div style='font-size:.8rem;color:#7FF5D8;margin-bottom:.3rem;'><b>Key Ingredients:</b> {ingr_str}</div>
          <div style='font-size:.8rem;color:#9090A0;'>{prod['why']}</div>
        </div>""", unsafe_allow_html=True)

    if rec["avoid"]:
        st.markdown("<div class='section-title'>🚫 Bahan yang Harus Dihindari</div>", unsafe_allow_html=True)
        st.markdown("".join(f"<span class='badge badge-pink'>❌ {a}</span>" for a in rec["avoid"]),
                    unsafe_allow_html=True)

    if rec["tips"]:
        st.markdown("<div class='section-title'>💡 Tips Perawatan</div>", unsafe_allow_html=True)
        for tip in rec["tips"]:
            st.markdown(f"<div class='tip-box'>✅ {tip}</div>", unsafe_allow_html=True)

    st.markdown("""<hr class='divider'>
    <div class='warn-box'>⚠️ <b>Disclaimer:</b> Hanya untuk edukasi. Konsultasikan dengan dokter kulit.</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: INFO MODEL
# ══════════════════════════════════════════════════════════════
elif PAGE == "Info Model":
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>📊 Informasi Model ML</div>", unsafe_allow_html=True)
    models = load_models()
    if models is None:
        st.error("❌ Model belum dilatih! Jalankan: `python src/train_model.py`")
        st.stop()

    results   = models["results"]
    best_name = models["best_name"]
    best      = results[best_name]
    class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    st.markdown(f"""
    <div class='card' style='border-color:#43D9AD;'>
      <div style='font-size:.8rem;color:#43D9AD;font-weight:600;'>🏆 BEST MODEL</div>
      <div style='font-size:1.5rem;font-weight:700;color:#E0E0FF;'>{best_name}</div>
      <div style='display:flex;gap:2rem;margin-top:.8rem;flex-wrap:wrap;'>
        <div><div style='font-size:1.3rem;font-weight:700;color:#6C63FF;'>{best['accuracy']:.4f}</div>
             <div style='font-size:.75rem;color:#6060A0;'>Accuracy</div></div>
        <div><div style='font-size:1.3rem;font-weight:700;color:#FF6584;'>{best['f1']:.4f}</div>
             <div style='font-size:.75rem;color:#6060A0;'>F1-Score</div></div>
        <div><div style='font-size:1.3rem;font-weight:700;color:#43D9AD;'>{best['cv_mean']:.4f}</div>
             <div style='font-size:.75rem;color:#6060A0;'>CV 5-Fold</div></div>
        <div><div style='font-size:1.3rem;font-weight:700;color:#FFB347;'>{best.get('auc') or 0:.4f}</div>
             <div style='font-size:.75rem;color:#6060A0;'>ROC-AUC</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Perbandingan Model</div>", unsafe_allow_html=True)
    st.plotly_chart(metrics_comparison_chart(results), use_container_width=True)

    st.markdown("<div class='section-title'>🔲 Confusion Matrix</div>", unsafe_allow_html=True)
    sel_model = st.selectbox("Pilih Model:", list(results.keys()), label_visibility="collapsed")
    st.plotly_chart(confusion_matrix_plotly(results[sel_model]["conf_matrix"], class_labels, sel_model),
                    use_container_width=True)

    if "Random Forest" in results:
        st.markdown("<div class='section-title'>🔑 Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
        rf = results["Random Forest"]["model"]
        st.plotly_chart(feature_importance_plotly(models["features"], rf.feature_importances_.tolist()),
                        use_container_width=True)

    st.markdown("<div class='section-title'>📂 Distribusi Dataset</div>", unsafe_allow_html=True)
    df = load_dataset()
    vc = df["diagnosis_name"].value_counts()
    fig_dist = go.Figure(go.Bar(
        x=vc.values.tolist(), y=vc.index.tolist(), orientation="h",
        marker=dict(color=["#6C63FF","#FF6584","#43D9AD","#FFB347","#A0C4FF","#B5EAD7"]),
        text=vc.values.tolist(), textposition="outside",
    ))
    fig_dist.update_layout(paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
                           font=dict(color="#E0E0FF",size=11),
                           xaxis=dict(gridcolor="#2D2D44"),
                           margin=dict(l=10,r=40,t=20,b=20), height=280)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("<div class='section-title'>📋 Tabel Metrik Lengkap</div>", unsafe_allow_html=True)
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
