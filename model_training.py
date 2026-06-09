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
    print("XGBoost tidak tersedia, skip model ini.")


DATA_FILE = "diabetes.csv"

DATA_URLS = [
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
    "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/diabetes.csv",
]

ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']



def load_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        for url in DATA_URLS:
            try:
                urllib.request.urlretrieve(url, DATA_FILE)
                print("Downloaded from:", url)
                break
            except Exception as e:
                print("Failed:", e)

    return pd.read_csv(DATA_FILE)



def preprocess(df):
    print("\nDataset shape:", df.shape)
    print("Target distribution:\n", df["Outcome"].value_counts())

    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



def make_pipeline(model):
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])



def get_models():
    models = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
            {"model__C": [0.1, 1, 10]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(random_state=42)),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [4, 6, 8],
            },
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = (
            make_pipeline(XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1
            )),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.05, 0.1],
            },
        )

    return models



def tune_threshold(y_true, y_prob):
    best_t, best_f1, best_metrics = 0.5, 0, {}

    for t in np.arange(0.2, 0.81, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = round(t, 2)

            best_metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1,
                "roc_auc": roc_auc_score(y_true, y_prob),
                "cm": confusion_matrix(y_true, y_pred),
            }

    return best_t, best_metrics



def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = get_models()

    best_model = None
    best_score = 0
    best_name = ""
    best_threshold = 0.5
    best_metrics = {}

    print("\n=== TRAINING START ===\n")

    for name, (pipe, grid) in models.items():
        print(f"Tuning {name}...")

        try:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
        except Exception as e:
            print("GridSearch error:", e)
            model = pipe.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        threshold, metrics = tune_threshold(y_test, y_prob)

        print(f"{name} | F1: {metrics['f1']:.4f} | Th: {threshold}")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model
            best_name = name
            best_threshold = threshold
            best_metrics = metrics

    return best_model, best_name, best_threshold, best_metrics


def save(model, name, threshold, metrics):
    os.makedirs("artifacts", exist_ok=True)

    # 🔥 IMPORTANT: tetap pisahkan seperti versi lama
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model.named_steps["model"], f)

    with open("artifacts/scaler.pkl", "wb") as f:
        pickle.dump(model.named_steps["scaler"], f)

    with open("artifacts/imputer.pkl", "wb") as f:
        pickle.dump(model.named_steps["imputer"], f)

    with open("artifacts/threshold.txt", "w") as f:
        f.write(str(threshold))

    metrics_out = {
        "model_name": name,
        "threshold": threshold,
        "accuracy": round(metrics["accuracy"] * 100, 2),
        "precision": round(metrics["precision"] * 100, 2),
        "recall": round(metrics["recall"] * 100, 2),
        "f1": round(metrics["f1"] * 100, 2),
        "roc_auc": round(metrics["roc_auc"] * 100, 2),
        "cm": metrics["cm"].tolist(),
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nSaved successfully in /artifacts")



if __name__ == "__main__":
    model, name, threshold, metrics = train()

    print("\nBEST MODEL:", name)
    print("THRESHOLD:", threshold)

    save(model, name, threshold, metrics)

    print("\nRun Streamlit: streamlit run app.py")