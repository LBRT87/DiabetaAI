# 🩸 DiabetaAI — Early Diabetes Risk Prediction

Machine Learning application for early diabetes risk prediction using a **Random Forest Classifier** and the **PIMA Indians Diabetes Dataset**.

---

# 📁 Project Structure

```bash
diabetes_project/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── diabetes.csv
├── model.pkl
├── scaler.pkl
├── imputer.pkl
└── threshold.txt
```

---

# ⚙️ Project Setup

## 1. Create Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

If successful, your terminal will show:

```bash
(venv)
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Run Model Training

Run the model training before starting Streamlit.

```bash
python train.py
```

The following files will be generated automatically:

- `model.pkl`
- `scaler.pkl`
- `imputer.pkl`
- `threshold.txt`

---

# 🚀 Run Streamlit App

After training is completed:

```bash
streamlit run app.py
```

Open in browser:

```bash
http://localhost:8501
```

---

# 👥 Team

- Airell Brandon Kho
- Elbert Joan
- Glenn Nielsen
- Marcell Kurniawan

BINUS University · Machine Learning · LA84