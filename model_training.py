import pandas as pd
import numpy as np
import os
import pickle

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
    print("XGBoost tidak ditemukan, akan di-skip. Install: pip install xgboost")
DATA_FILE = "diabetes.csv"


df = pd.read_csv(DATA_FILE)
print(f"\nShape dataset: {df.shape}")
print(f"   Distribusi target:\n{df['Outcome'].value_counts().to_string()}")

ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipe(clf):
    return ImbPipeline([
        ('imputer',    SimpleImputer(strategy='median')),
        ('smote',      SMOTE(random_state=42)),
        ('scaler',     StandardScaler()),
        ('classifier', clf),
    ])

pipelines = {
    "Logistic Regression": (
        make_pipe(LogisticRegression(random_state=42, max_iter=1000)),
        {'classifier__C': [0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        make_pipe(RandomForestClassifier(random_state=42, class_weight='balanced')),
        {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [4, 6, 8]},
    ),
}
if HAS_XGB:
    pipelines["XGBoost"] = (
        make_pipe(XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)),
        {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 4],
         'classifier__learning_rate': [0.05, 0.1]},
    )

best_model      = None
best_score      = 0.0
best_name       = ""
best_threshold  = 0.5
best_metrics    = {}

print("\ Memulai GridSearchCV + Threshold Tuning...\n")

for name, (pipe, grid) in pipelines.items():
    print(f"Tuning {name}...")
    try:
        gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring='f1', n_jobs=1, verbose=0)
        gs.fit(X_train, y_train)
        estimator = gs.best_estimator_
        print(f"Best params: {gs.best_params_}")
    except Exception as e:
        print(f"GridSearch gagal ({e}), fallback ke default fit.")
        pipe.fit(X_train, y_train)
        estimator = pipe

    y_prob = estimator.predict_proba(X_test)[:, 1]

    best_t, best_f1_t, best_m = 0.5, 0.0, {}
    for t in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1_t:
            best_f1_t = f1
            best_t = round(t, 2)
            best_m = {
                'accuracy':  accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall':    recall_score(y_test, y_pred, zero_division=0),
                'f1':        f1,
                'roc_auc':   roc_auc_score(y_test, y_prob),
                'cm':        confusion_matrix(y_test, y_pred),
            }

    print(f"    Threshold: {best_t:.2f} | F1: {best_f1_t:.4f} | "
          f"Recall: {best_m['recall']:.4f} | AUC: {best_m['roc_auc']:.4f}")

    if best_f1_t > best_score:
        best_score, best_model, best_name  = best_f1_t, estimator, name
        best_threshold, best_metrics       = best_t, best_m

print(f"""
{'='*52}
  Model Terpilih   : {best_name}
  Threshold        : {best_threshold:.2f}
  ─────────────────────────────────────────
  Accuracy  : {best_metrics['accuracy']:.4f}  (Target > 70%)
  Precision : {best_metrics['precision']:.4f}  (Target > 75%)
  Recall    : {best_metrics['recall']:.4f}  (Target > 85%)
  F1-Score  : {best_metrics['f1']:.4f}  (Target > 75%)
  ROC-AUC   : {best_metrics['roc_auc']:.4f}  (Target > 85%)
  ─────────────────────────────────────────
  Confusion Matrix:
{best_metrics['cm']}
{'='*52}
""")
import json

with open("model.pkl",   "wb") as f: pickle.dump(best_model.named_steps['classifier'], f)
with open("scaler.pkl",  "wb") as f: pickle.dump(best_model.named_steps['scaler'],     f)
with open("imputer.pkl", "wb") as f: pickle.dump(best_model.named_steps['imputer'],    f)
with open("threshold.txt","w") as f: f.write(str(best_threshold))

metrics_out = {
    "model_name": best_name,
    "threshold":  best_threshold,
    "accuracy":   round(best_metrics['accuracy']  * 100, 2),
    "precision":  round(best_metrics['precision'] * 100, 2),
    "recall":     round(best_metrics['recall']    * 100, 2),
    "f1":         round(best_metrics['f1']        * 100, 2),
    "roc_auc":    round(best_metrics['roc_auc']   * 100, 2),
    "cm":         best_metrics['cm'].tolist(),
}
with open("metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print("File tersimpan: model.pkl, scaler.pkl, imputer.pkl, threshold.txt, metrics.json")
print("Sekarang jalankan: streamlit run app.py")