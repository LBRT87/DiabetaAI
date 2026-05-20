import numpy as np
import pandas as pd


_MEDICAL_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def predict_diabetes(model, scaler, imputer, threshold, input_data: dict):
    raw = pd.DataFrame({
        "Pregnancies":              [input_data["pregnancies"]],
        "Glucose":                  [input_data["glucose"]],
        "BloodPressure":            [input_data["blood_pressure"]],
        "SkinThickness":            [input_data["skin_thickness"]],
        "Insulin":                  [input_data["insulin"]],
        "BMI":                      [input_data["bmi"]],
        "DiabetesPedigreeFunction": [input_data["dpf"]],
        "Age":                      [input_data["age"]],
    })

    raw[_MEDICAL_ZERO_COLS] = raw[_MEDICAL_ZERO_COLS].replace(0, np.nan)

    imputed   = pd.DataFrame(imputer.transform(raw), columns=raw.columns)
    scaled    = scaler.transform(imputed)
    prob      = model.predict_proba(scaled)[0][1]
    prediction = int(prob >= threshold)

    return prediction, prob, imputed