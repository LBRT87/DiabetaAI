# 🩸 DiabetaAI — Early Diabetes Risk Prediction

**BINUS University · COMP6577001 Machine Learning · LA84**

> Early Diabetes Risk Prediction System menggunakan Random Forest Classifier
> berbasis dataset PIMA Indians Diabetes.

---

## 📁 Struktur Folder

```
diabetes_project/
│
├── app.py              ← Aplikasi Streamlit (frontend + backend)
├── train.py            ← Script training model
├── requirements.txt    ← Dependensi Python
├── .gitignore
├── README.md
│
└── (setelah train.py dijalankan)
    ├── diabetes.csv    ← Dataset (auto-download)
    ├── model.pkl       ← Model Random Forest terlatih
    ├── scaler.pkl      ← StandardScaler terlatih
    ├── imputer.pkl     ← SimpleImputer terlatih
    └── threshold.txt   ← Optimal decision threshold
```

---

## ⚙️ Setup & Installation

### 1. Prasyarat

| Item | Versi |
|---|---|
| Python | **3.10 atau 3.11** (direkomendasikan) |
| pip | terbaru |
| Git | terbaru |

> ⚠️ **Hindari Python 3.12+ atau 3.14** — beberapa dependensi (XGBoost, imbalanced-learn)
> belum sepenuhnya stabil di versi tersebut.  
> Cek versi Python: `python --version`

---

### 2. Clone / Siapkan Folder

Jika menggunakan Git:
```bash
git clone https://github.com/<username>/diabetes_project.git
cd diabetes_project
```

Atau cukup taruh semua file di satu folder, lalu buka terminal di folder tersebut.

---

### 3. Buat Virtual Environment (venv)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Setelah aktif, prompt terminal akan berubah menjadi `(venv) ...`

---

### 4. Install Dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Proses ini membutuhkan koneksi internet dan sekitar 2–5 menit tergantung kecepatan jaringan.

---

### 5. Training Model

```bash
python train.py
```

Script ini akan:
1. Download `diabetes.csv` secara otomatis (dari GitHub)
2. Melakukan preprocessing (imputer, SMOTE, scaler)
3. Melatih 3 model (Logistic Regression, Random Forest, XGBoost) dengan GridSearchCV
4. Memilih model terbaik berdasarkan F1-Score
5. Menyimpan `model.pkl`, `scaler.pkl`, `imputer.pkl`, `threshold.txt`

---

### 6. Jalankan Aplikasi

```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

---

## 🚀 Deployment

### Opsi A: Streamlit Community Cloud (Gratis, Direkomendasikan)

1. Push ke GitHub (pastikan `model.pkl`, `scaler.pkl`, `imputer.pkl` ikut di-commit)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Set **Main file path:** `app.py`
5. Deploy → dapat link publik

> 💡 **Tips:** Karena `.gitignore` mengecualikan `.pkl`, sebelum push ke GitHub,
> hapus baris `*.pkl` dari `.gitignore` agar file model ikut ter-commit.

### Opsi B: Local Demo

Cukup jalankan `streamlit run app.py` dan share screen saat demo.

---

## 📊 Hasil Model

| Metrik | Target | Pencapaian |
|---|---|---|
| Accuracy | > 70% | **77.92%** ✅ |
| Precision | > 75% | **76.47%** ✅ |
| Recall | > 85% | **88.89%** ✅ |
| F1-Score | > 75% | **82.11%** ✅ |
| ROC-AUC | > 85% | **90.17%** ✅ |

---

## 👥 Tim

| Nama | NIM |
|---|---|
| Airell Brandon Kho | 2902577260 |
| Elbert Joan | 2902592135 |
| Glenn Nielsen | 2902590086 |
| Marcell Kurniawan | 2902591164 |

**Dosen Pembimbing:** Dr. Maria Susan Anggreainy, S.Kom., M.Kom.

---

## 📚 Referensi

1. [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. Smith et al. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
3. Sisodia & Sisodia (2018). *Prediction of Diabetes using Classification Algorithms.* Procedia CS, 132.
4. Kavakiotis et al. (2017). *Machine Learning and Data Mining Methods in Diabetes Research.* CSBJ, 15.
