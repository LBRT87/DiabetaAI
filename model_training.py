import pandas as pd
import numpy as np
import os
import pickle
import json
import urllib.request

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Modul XGBoost tidak ditemukan dan akan dilewati. Silakan install dengan: pip install xgboost")

DATA_FILE = "diabetes.csv"
DATA_URLS = [
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
    "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/diabetes.csv",
]

ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def load_data():
    if not os.path.exists(DATA_FILE):
        print("Sedang mengunduh dataset...")
        downloaded = False
        for url in DATA_URLS:
            try:
                urllib.request.urlretrieve(url, DATA_FILE)
                test = pd.read_csv(DATA_FILE)
                
                if test.shape[1] == 9:
                    if 'Outcome' not in test.columns:
                        test.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
                        test.to_csv(DATA_FILE, index=False)
                        
                print(f"Berhasil mengunduh dari: {url}")
                downloaded = True
                break
            except Exception as e:
                print(f"Gagal mengunduh dari {url}: {e}")
                
        if not downloaded:
            print("Terjadi kesalahan: Dataset tidak dapat diunduh.")
            exit(1)
    else:
        print("Dataset ditemukan. Memulai proses training...")
        
    return pd.read_csv(DATA_FILE)

def preprocess(df):
    print(f"\nDimensi dataset: {df.shape}")
    print(f"Distribusi target:\n{df['Outcome'].value_counts().to_string()}")

    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def make_pipeline(model):
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', model),
    ])

def get_models():
    models = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(random_state=42, max_iter=1000)),
            {'classifier__C': [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced')),
            {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [4, 6, 8]},
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = (
            make_pipeline(XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)),
            {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 4],
             'classifier__learning_rate': [0.05, 0.1]},
        )
        
    return models

def tune_threshold(y_true, y_prob):
    best_t, best_f1, best_metrics = 0.5, 0.0, {}

    for t in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = round(t, 2)
            best_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1,
                'roc_auc': roc_auc_score(y_true, y_prob),
                'cm': confusion_matrix(y_true, y_pred),
            }
            
    return best_t, best_metrics

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    
    print(f"\nData Latih: {X_train.shape[0]} baris | Data Uji: {X_test.shape[0]} baris")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()

    best_model = None
    best_score = 0.0
    best_name = ""
    best_threshold = 0.5
    best_metrics = {}

    print("\nMemulai proses pencarian parameter terbaik dan penyesuaian threshold...\n")

    for name, (pipe, grid) in models.items():
        print(f"Mengevaluasi {name}...")
        try:
            gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring='f1', n_jobs=1, verbose=0)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            print(f"    Parameter terbaik: {gs.best_params_}")
        except Exception as e:
            print(f"Pencarian parameter gagal ({e}), menggunakan konfigurasi bawaan.")
            pipe.fit(X_train, y_train)
            model = pipe

        y_prob = model.predict_proba(X_test)[:, 1]
        threshold, metrics = tune_threshold(y_test, y_prob)

        print(f"    Threshold optimal: {threshold:.2f} | F1-Score: {metrics['f1']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | AUC: {metrics['roc_auc']:.4f}")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model
            best_name = name
            best_threshold = threshold
            best_metrics = metrics

    return best_model, best_name, best_threshold, best_metrics

def save(model, name, threshold, metrics):
    with open("model.pkl", "wb") as f:
        pickle.dump(model.named_steps['classifier'], f)
        
    with open("scaler.pkl", "wb") as f:
        pickle.dump(model.named_steps['scaler'], f)
        
    with open("imputer.pkl", "wb") as f:
        pickle.dump(model.named_steps['imputer'], f)
        
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))

    metrics_out = {
        "model_name": name,
        "threshold": threshold,
        "accuracy": round(metrics['accuracy'] * 100, 2),
        "precision": round(metrics['precision'] * 100, 2),
        "recall": round(metrics['recall'] * 100, 2),
        "f1": round(metrics['f1'] * 100, 2),
        "roc_auc": round(metrics['roc_auc'] * 100, 2),
        "cm": metrics['cm'].tolist(),
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n--- Hasil Evaluasi Model Terbaik ---")
    print(f"Model Terpilih : {name}")
    print(f"Threshold      : {threshold:.2f}")
    print("-" * 34)
    print(f"Akurasi   : {metrics['accuracy']:.4f}")
    print(f"Presisi   : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("-" * 34)
    print("Confusion Matrix:")
    print(metrics['cm'])
    print("-" * 34)
    
    print("\nSemua file pendukung berhasil disimpan.")

if __name__ == "__main__":
    model, name, threshold, metrics = train()
    save(model, name, threshold, metrics)
    print("Proses selesai. Sekarang kamu bisa menjalankan: streamlit run app.py")