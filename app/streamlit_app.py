"""
streamlit_app.py  —  Skincare Recommender (Clinical / Elegant Light Theme)
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
    page_title="DermAI | Clinical Skincare Recommender",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide sidebar initially to focus on tabs
)

# ── Global CSS (Clean Medical / Light Theme) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}

/* ── Background ── */
.stApp { 
    background-color: #F8FAFC; 
    color: #334155; 
}

/* ── Hide top padding somewhat ── */
.block-container {
    padding-top: 2rem !important;
}

/* ── Headers & Titles ── */
h1, h2, h3 { color: #0F172A !important; font-weight: 700 !important; }

/* ── Cards ── */
.clinical-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.clinical-card:hover { 
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); 
}

.prescription-card {
    background: #F0FDF4;
    border-left: 4px solid #22C55E;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.warning-card {
    background: #FEF2F2;
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    color: #991B1B;
}

/* ── Hero ── */
.hero-title {
    font-size: 3rem; 
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.hero-sub { 
    font-size: 1.2rem; 
    color: #64748B; 
    font-weight: 400;
    margin-bottom: 2.5rem; 
}

/* ── Badges / Pills ── */
.badge {
    display: inline-block; padding: 0.35rem 0.8rem;
    border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    margin: 0.2rem;
}
.badge-blue  { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.badge-green { background: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }
.badge-red   { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
.badge-gray  { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }

/* ── Section headings ── */
.section-title {
    font-size: 1.35rem; font-weight: 700;
    color: #1E293B; margin: 2rem 0 1rem;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 0.5rem;
}

/* ── Custom Buttons ── */
.stButton > button {
    background-color: #2563EB;
    color: white; 
    border: none; 
    border-radius: 8px;
    padding: 0.6rem 2rem; 
    font-weight: 600; 
    font-size: 1rem;
    transition: background-color 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

/* ── Streamlit Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    color: #64748B;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 3px solid #2563EB !important;
}

/* ── Form styling ── */
div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource(show_spinner="Memuat model medis...")
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
# MAIN NAVIGATION (TABS)
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='hero-title'>⚕️ DermAI Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Clinical Grade Skincare Recommendations & Dermatological Analysis</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 Tinjauan Sistem", "🔬 Diagnosis & Rekomendasi", "📊 Analisis Data Klinis"])

# ══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="clinical-card">
        <h3 style='margin-top:0; color:#0F172A;'>Selamat Datang di DermAI</h3>
        <p style='color:#475569; line-height:1.7;'>
            DermAI adalah sistem rekomendasi berbasis *Machine Learning* yang memadukan keakuratan prediksi algoritma diagnostik dengan *Expert System* perawatan kulit dermatologis.
            Sistem ini dilatih menggunakan ratusan data rekam medis pasien untuk mendeteksi kondisi kulit secara presisi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="clinical-card" style="text-align:center;">
        <div style="font-size:2.5rem;">🏥</div>
        <div style="font-weight:700; font-size:1.1rem; margin-top:0.5rem;">Diagnosis Presisi</div>
        <div style="font-size:0.9rem; color:#64748B;">Menganalisis 34 parameter klinis untuk memprediksi 6 penyakit kulit spesifik.</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("""
    <div class="clinical-card" style="text-align:center;">
        <div style="font-size:2.5rem;">🧬</div>
        <div style="font-weight:700; font-size:1.1rem; margin-top:0.5rem;">Profil Biometrik</div>
        <div style="font-size:0.9rem; color:#64748B;">Mengidentifikasi 5 tipe profil kulit bawaan menggunakan K-Means Clustering.</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown("""
    <div class="clinical-card" style="text-align:center;">
        <div style="font-size:2.5rem;">💊</div>
        <div style="font-weight:700; font-size:1.1rem; margin-top:0.5rem;">Rekomendasi Medis</div>
        <div style="font-size:0.9rem; color:#64748B;">Memfilter bahan iritan dan menyarankan produk komersial yang tervalidasi aman.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Cakupan Diagnosis Medis Terdaftar</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:1rem;'>Sistem AI kami dilatih untuk membedakan 6 kondisi dermatologis spesifik dengan menganalisis pola dari 34 fitur klinis dan histopatologis.</p>", unsafe_allow_html=True)
    
    with st.expander("📚 Ensiklopedia Kondisi Kulit (Klik untuk detail)"):
        st.markdown("""
        * **Psoriasis**: Penyakit autoimun kronis yang mempercepat siklus pertumbuhan sel kulit, menyebabkan penumpukan sel mati bersisik tebal, berwarna perak, dan bercak merah yang gatal atau perih.
        * **Seboreic Dermatitis**: Kondisi kulit umum yang terutama memengaruhi kulit kepala dan area berminyak lainnya (seperti wajah atau dada), menyebabkan kulit bersisik, merah, dan ketombe membandel. Diduga berkaitan dengan jamur Malassezia.
        * **Lichen Planus**: Kondisi peradangan yang dapat memengaruhi kulit dan selaput lendir (seperti mulut). Ditandai dengan benjolan keunguan yang gatal, datar, dan sering berbentuk poligonal (banyak sudut).
        * **Pityriasis Rosea**: Ruam kulit yang biasanya dimulai dengan satu bercak besar berbentuk oval (herald patch) yang kemudian diikuti oleh bercak-bercak lebih kecil di dada, perut, atau punggung. Bersifat sementara dan sering sembuh sendiri.
        * **Chronic Dermatitis**: Istilah umum untuk peradangan kulit kronis (seperti eksim) yang ditandai dengan kulit kering, sangat gatal, merah, dan menebal karena sering digaruk akibat iritasi berkepanjangan.
        * **Pityriasis Rubra Pilaris**: Gangguan langka yang menyebabkan bercak merah bersisik menetap (hiperkeratosis), penebalan kulit pada telapak tangan dan kaki, serta benjolan kecil di sekitar folikel rambut.
        """)

    with st.expander("🧬 Kamus Gejala Klinis (Glossary)"):
        st.markdown("""
        * **Erythema (Eritema)**: Kemerahan pada kulit akibat peningkatan aliran darah pada kapiler yang meradang.
        * **Scaling (Deskuamasi)**: Pengelupasan lapisan terluar kulit, sering disebut sebagai kulit bersisik.
        * **Koebner Phenomenon**: Munculnya lesi/ruam kulit baru pada area yang sebelumnya mengalami trauma fisik (seperti goresan atau bekas luka).
        * **Polygonal Papules**: Benjolan kecil padat di permukaan kulit yang batasnya memiliki banyak sudut (tidak bulat sempurna).
        * **Follicular Papules**: Benjolan kecil yang terbentuk tepat di lokasi folikel rambut (pori-pori rambut).
        """)

    st.markdown("""
    <br>
    <div class='warning-card'>
      <b>Pemberitahuan Medis:</b><br>
      Aplikasi ini dikembangkan untuk eksperimen AI dan edukasi. Hasil diagnosis <b>tidak dapat diandalkan sebagai pengganti pemeriksaan dokter kulit berlisensi (Sp.KK/Sp.DVE)</b>. Diagnosis final dan peresepan obat keras harus dilakukan oleh dokter.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: ANALYSIS & RECOMMENDATION
# ══════════════════════════════════════════════════════════════
with tab2:
    models = load_models()
    if models is None:
        st.error("Sistem gagal memuat model klinis. Pastikan Anda telah menjalankan proses training (train_model.py).")
    else:
        st.markdown("""
        <div style="margin-bottom: 1rem; color: #475569;">
            Silakan masukkan parameter klinis pasien melalui kuesioner berikut untuk memulai proses diagnosis otomatis.
        </div>
        """, unsafe_allow_html=True)

        with st.form("clinical_form"):
            st.markdown("<h4 style='color:#1E293B; margin-bottom:1rem;'>Demografi & Riwayat</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            age = col1.number_input("Usia Pasien", min_value=1, max_value=100, value=25, step=1)
            family_hist = col2.selectbox("Riwayat Keluarga (Kondisi Dermatologis)", [0, 1], format_func=lambda x: "Ada (Positif)" if x else "Tidak Ada (Negatif)")

            st.markdown("<h4 style='color:#1E293B; margin-top:2rem; margin-bottom:1rem;'>Pemeriksaan Klinis (Intensitas 0-3)</h4>", unsafe_allow_html=True)
            
            clinical_symptoms = [
                "erythema","scaling","definite_borders","itching",
                "koebner_phenomenon","polygonal_papules","follicular_papules",
                "oral_mucosal_involvement","knee_elbow_involvement","scalp_involvement",
            ]
            labels = {
                "erythema": "Eritema (Kemerahan)", "scaling": "Deskuamasi (Bersisik)",
                "definite_borders": "Batas Lesi Tegas", "itching": "Pruritus (Gatal)",
                "koebner_phenomenon": "Fenomena Koebner", "polygonal_papules": "Papul Poligonal",
                "follicular_papules": "Papul Folikuler", "oral_mucosal_involvement": "Lesi Mukosa Oral",
                "knee_elbow_involvement": "Lesi Ekstensor (Lutut/Siku)", "scalp_involvement": "Lesi Skalp (Kepala)",
            }
            
            vals = {}
            r1, r2 = st.columns(2)
            for i, feat in enumerate(clinical_symptoms):
                col = r1 if i % 2 == 0 else r2
                vals[feat] = col.select_slider(labels[feat], options=[0, 1, 2, 3], value=0)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Jalankan Diagnosis AI")

        if submitted:
            st.markdown("<div class='section-title'>Laporan Medis DermAI</div>", unsafe_allow_html=True)
            
            # Predict Logic
            all_features_ordered = [
                "erythema","scaling","definite_borders","itching","koebner_phenomenon","polygonal_papules","follicular_papules",
                "oral_mucosal_involvement","knee_elbow_involvement","scalp_involvement","family_history","melanin_incontinence","eosinophils_in_infiltrate","PNL_infiltrate",
                "fibrosis_papillary_dermis","exocytosis","acanthosis","hyperkeratosis","parakeratosis","clubbing_rete_ridges","elongation_rete_ridges",
                "thinning_suprapapillary_epidermis","spongiform_pustule","munro_microabcess","focal_hypergranulosis","disappearance_granular_layer","vacuolisation_damage_basal_layer",
                "spongiosis","saw_tooth_appearance","follicular_horn_plug","perifollicular_parakeratosis","inflammatory_mononuclear_infiltrate","band_like_infiltrate","age",
            ]
            raw = {f: 0.0 for f in all_features_ordered}
            for feat in clinical_symptoms: raw[feat] = float(vals[feat])
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
            conf_score = max(proba) * 100

            rec = get_recommendation(cond_name, cluster_id, age)
            skin_prof = rec["skin_profile"]

            # DISPLAY RESULTS (CLINICAL REPORT STYLE)
            st.markdown("<p style='color:#64748B; font-size:0.9rem; margin-bottom:1.5rem;'>Di bawah ini adalah kalkulasi diagnostik yang dihasilkan oleh algoritma <i>Random Forest Classifier</i> berdasarkan data gejala yang Anda masukkan.</p>", unsafe_allow_html=True)

            colA, colB = st.columns([1.5, 1])
            with colA:
                st.markdown(f"""
                <div class="clinical-card" style="border-top: 4px solid #2563EB;">
                    <div style="font-size:0.85rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Diagnosis Utama (Primary Diagnosis)</div>
                    <div style="font-size:2rem; color:#0F172A; font-weight:800; margin-top:0.2rem;">{cond_name}</div>
                    <div style="margin-top:0.5rem; margin-bottom:1rem;">
                        <span class="badge badge-blue">Confidence: {conf_score:.1f}%</span>
                        <span class="badge badge-gray">Usia Pasien: {age} thn</span>
                    </div>
                    <div style="font-size:0.9rem; color:#475569;"><b>Probabilitas Diferensial (Top 3):</b></div>
                """, unsafe_allow_html=True)
                
                # Show top 3 predictions
                top_indices = np.argsort(proba)[::-1][:3]
                class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
                for idx in top_indices:
                    prob_val = proba[idx] * 100
                    if prob_val > 0.1:
                        st.markdown(f"<div style='font-size:0.85rem; margin-top:0.2rem;'>• {class_labels[idx]}: <b>{prob_val:.1f}%</b></div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                if rec["avoid"]:
                    avoid_html = "".join([f"<li style='margin-bottom:0.3rem;'>{a}</li>" for a in rec["avoid"]])
                    st.markdown(f"""
                    <div class="warning-card">
                        <strong style="display:block; margin-bottom:0.5rem; font-size:1.1rem;">⚠️ Kontraindikasi & Bahan Iritan (Sangat Dihindari)</strong>
                        <div style="font-size:0.85rem; margin-bottom:0.5rem; color:#991B1B;">Bahan-bahan di bawah ini berpotensi merusak <i>skin barrier</i> atau memicu respon inflamasi (<i>flare-up</i>) pada kondisi {cond_name}:</div>
                        <ul style="margin:0; padding-left:1.2rem; font-size:0.9rem;">{avoid_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)

            with colB:
                st.markdown(f"""
                <div class="clinical-card" style="border-top: 4px solid #10B981; height: 100%;">
                    <div style="font-size:0.85rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Profil Biometrik Kulit</div>
                    <div style="font-size:1.6rem; color:#0F172A; font-weight:700; margin-top:0.2rem;">{skin_prof['icon']} {skin_prof['name']}</div>
                    <p style="color:#475569; font-size:0.9rem; line-height:1.6; margin-top:0.8rem;">
                        Berdasarkan pemrosesan <i>K-Means Clustering</i>, kondisi Anda masuk ke dalam klaster <b>{skin_prof['name']}</b>.<br><br>
                        <b>Karakteristik Umum:</b><br>{skin_prof['desc']}
                    </p>
                    <div style="margin-top:1rem;">
                        <span class="badge badge-green">{rec['concern']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Protokol Perawatan (Rekomendasi Skincare)</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.9rem; color:#64748B;'>Berdasarkan diagnosis <b>{}</b> dan klaster kulit <b>{}</b>, <i>Rule-Based Expert System</i> kami memformulasikan rutinitas *skincare* yang aman dan terfokus pada perbaikan kondisi.</p>".format(cond_name, skin_prof['name']), unsafe_allow_html=True)
            
            cat_map = {
                "cleanser": ("Pembersih Wajah", "🧼"),
                "toner": ("Toner / Essence", "💧"),
                "serum": ("Serum Aktif", "✨"),
                "moisturizer": ("Pelembap (Emolien)", "🧴"),
                "sunscreen": ("Proteksi UV", "☀️")
            }

            for cat in ["cleanser", "toner", "serum", "moisturizer", "sunscreen"]:
                if cat in rec["products"]:
                    p = rec["products"][cat]
                    cat_name, icon = cat_map[cat]
                    st.markdown(f"""
                    <div class="prescription-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span style="font-weight:700; color:#166534; font-size:1.1rem;">{icon} {cat_name}</span>
                            <span style="font-size:0.8rem; color:#15803D; font-weight:600; background:#DCFCE7; padding:0.2rem 0.6rem; border-radius:4px;">{p['brand']}</span>
                        </div>
                        <div style="font-size:1.25rem; font-weight:700; color:#064E3B; margin-bottom:0.5rem;">{p['name']}</div>
                        <div style="font-size:0.9rem; color:#166534; margin-bottom:0.5rem;"><b>Bahan Aktif:</b> {', '.join(p['ingredients'])}</div>
                        <div style="font-size:0.9rem; color:#3F6212; font-style:italic;">"{p['why']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

            if rec["tips"]:
                st.markdown("<div class='section-title'>Protokol Gaya Hidup</div>", unsafe_allow_html=True)
                tips_html = "".join([f"<li style='margin-bottom:0.5rem; color:#334155;'>{t}</li>" for t in rec["tips"]])
                st.markdown(f"""
                <div class="clinical-card" style="background:#F8FAFC;">
                    <ul style="margin:0; padding-left:1.2rem;">{tips_html}</ul>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO & ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab3:
    models = load_models()
    if models is not None:
        results = models["results"]
        best_name = models["best_name"]
        best = results[best_name]
        class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

        st.markdown("<h3 style='color:#0F172A; margin-bottom:1.5rem;'>Performa Algoritma Diagnostik</h3>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='clinical-card'><b>Akurasi</b><br><span style='font-size:1.8rem; font-weight:800; color:#2563EB;'>{best['accuracy']:.2%}</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='clinical-card'><b>F1-Score</b><br><span style='font-size:1.8rem; font-weight:800; color:#10B981;'>{best['f1']:.2%}</span></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='clinical-card'><b>CV 5-Fold</b><br><span style='font-size:1.8rem; font-weight:800; color:#F59E0B;'>{best['cv_mean']:.2%}</span></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='clinical-card'><b>ROC-AUC</b><br><span style='font-size:1.8rem; font-weight:800; color:#8B5CF6;'>{best.get('auc', 0):.2%}</span></div>", unsafe_allow_html=True)

        st.markdown(f"<div style='margin-bottom:2rem;'><span class='badge badge-gray'>Model Utama: {best_name}</span></div>", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("<b>Perbandingan Antar Model</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.85rem; color:#64748B;'>Grafik batang ini mengkomparasi metrik Evaluasi (Akurasi, Precision, Recall, F1) untuk keempat algoritma yang diuji. Random Forest biasanya mendominasi karena kemampuannya menangani interaksi fitur yang kompleks (ensemble method).</p>", unsafe_allow_html=True)
            st.plotly_chart(metrics_comparison_chart(results), use_container_width=True)
        with colB:
            st.markdown("<b>Matriks Kebingungan (Confusion Matrix)</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.85rem; color:#64748B;'><i>Confusion Matrix</i> memetakan Prediksi AI vs Kenyataan (Aktual). Jika warna biru gelap terfokus di garis diagonal (kiri atas ke kanan bawah), itu menandakan tebakan model mayoritas benar tanpa ada kelas penyakit yang tertukar.</p>", unsafe_allow_html=True)
            sel_model = st.selectbox("Pilih Model:", list(results.keys()), label_visibility="collapsed")
            st.plotly_chart(confusion_matrix_plotly(results[sel_model]["conf_matrix"], class_labels, sel_model), use_container_width=True)

        if "Random Forest" in results:
            st.markdown("<b>Signifikansi Gejala (Feature Importance)</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.85rem; color:#64748B;'>Bagan ini menunjukkan bobot prioritas model AI. Semakin panjang batangnya, semakin penting gejala tersebut dalam mempengaruhi keputusan akhir diagnosis model (misalnya *Scaling* atau *Erythema* seringkali memegang porsi bobot yang besar).</p>", unsafe_allow_html=True)
            rf = results["Random Forest"]["model"]
            st.plotly_chart(feature_importance_plotly(models["features"], rf.feature_importances_.tolist()), use_container_width=True)
    else:
        st.warning("Model belum dilatih.")
