# Skincare Recommender — Personalized Skincare for Young Adults

Sistem rekomendasi produk skincare personal untuk anak muda berbasis **Classical Machine Learning**, menggunakan **UCI Dermatology Dataset**.

## 📌 Tech Stack
- **ML**: Random Forest, SVM, Decision Tree, Logistic Regression, K-Means
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit
- **Dataset**: [UCI Dermatology Dataset](https://archive.ics.uci.edu/dataset/33/dermatology) (ID: 33)

## 🗂️ Struktur Proyek
```
skincare-recommender/
├── data/          # Dataset cache (auto-generated)
├── models/        # Trained model artifacts (auto-generated)
├── plots/         # Visualization plots (auto-generated)
├── src/
│   ├── data_loader.py    # UCI data fetching & cleaning
│   ├── preprocessing.py  # Scaling, encoding, feature selection
│   ├── train_model.py    # Model training & evaluation
│   ├── evaluate.py       # Plotly visualization helpers
│   └── recommend.py      # Rule-based recommendation engine
├── app/
│   └── streamlit_app.py  # Main Streamlit web app
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Latih model (hanya perlu sekali)
```bash
python src/train_model.py
```

### 3. Jalankan aplikasi Streamlit
```bash
python -m streamlit run app\streamlit_app.py
streamlit run app/streamlit_app.py
```

## 📊 Dataset
- **Nama**: UCI Dermatology Dataset
- **Sumber**: https://archive.ics.uci.edu/dataset/33/dermatology
- **Instansi**: 366 | **Fitur**: 34 | **Kelas**: 6
- **Kondisi**: Psoriasis, Seboreic Dermatitis, Lichen Planus, Pityriasis Rosea, Chronic Dermatitis, Pityriasis Rubra Pilaris

## 📖 Citation
```
Ilter, N. & Guvenir, H. (1998). Dermatology [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5FK5P
```
