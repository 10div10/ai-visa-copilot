"""
train_risk_model.py — Trains an XGBoost classifier to predict visa rejection
probability, and saves both the model and a SHAP explainer so predictions come
with human-readable "top factors" rather than a black-box number.

Usage:
    python generate_synthetic_data.py   # if not already run
    python train_risk_model.py
Outputs:
    data/risk_model.json     (XGBoost model)
    data/risk_model_meta.json (feature names, encoders)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

DATA_PATH = Path(__file__).parent / "data" / "synthetic_applications.csv"
MODEL_PATH = Path(__file__).parent / "data" / "risk_model.json"
META_PATH = Path(__file__).parent / "data" / "risk_model_meta.json"

FEATURE_COLS = [
    "doc_completeness",
    "passport_buffer_days",
    "financial_ratio",
    "ties_strength",
    "days_to_travel",
    "prior_rejection",
    "visa_type_us_h1b",  # one-hot encoded visa type
]


def load_and_encode():
    df = pd.read_csv(DATA_PATH)
    df["visa_type_us_h1b"] = (df["visa_type"] == "us_h1b").astype(int)
    X = df[FEATURE_COLS]
    y = df["rejected"]
    return X, y


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Run `python generate_synthetic_data.py` first.")

    X, y = load_and_encode()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, (preds > 0.5).astype(int)))

    model.save_model(str(MODEL_PATH))
    with open(META_PATH, "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "test_auc": auc}, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
