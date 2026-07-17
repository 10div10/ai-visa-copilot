"""
risk_model.py — Loads the trained XGBoost risk model and scores a new application,
returning both a rejection probability and the top contributing factors (via SHAP)
in plain-language form, so the explanation is grounded in the actual model rather
than a generic template.
"""
import json
from pathlib import Path
from typing import TypedDict, List

import numpy as np
import xgboost as xgb
import shap

MODEL_PATH = Path(__file__).parent / "data" / "risk_model.json"
META_PATH = Path(__file__).parent / "data" / "risk_model_meta.json"

# Human-readable descriptions for each feature, used to build the explanation text.
FEATURE_LABELS = {
    "doc_completeness": "document completeness",
    "passport_buffer_days": "passport validity buffer",
    "financial_ratio": "financial proof strength",
    "ties_strength": "ties-to-home-country strength",
    "days_to_travel": "application timing relative to travel date",
    "prior_rejection": "prior visa rejection history",
    "visa_type_us_h1b": "visa type",
}


class ApplicationInput(TypedDict, total=False):
    visa_type: str                # "us_h1b" | "schengen_tourist"
    doc_completeness: float       # 0-1, fraction of required docs present
    passport_buffer_days: int     # days of passport validity beyond the minimum required
    financial_ratio: float        # available funds / required minimum
    ties_strength: float          # 0-1 proxy for job/property/family ties to home country
    days_to_travel: int           # days between application and intended travel/start date
    prior_rejection: int          # 0 or 1


class RiskResult(TypedDict):
    rejection_probability: float
    risk_level: str
    top_factors: List[dict]


class RiskModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "No trained model found. Run `python generate_synthetic_data.py` "
                "then `python train_risk_model.py` first."
            )
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(MODEL_PATH))
        with open(META_PATH) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.explainer = shap.TreeExplainer(self.model)

    def _vectorize(self, app: ApplicationInput) -> np.ndarray:
        row = {
            "doc_completeness": app.get("doc_completeness", 1.0),
            "passport_buffer_days": app.get("passport_buffer_days", 180),
            "financial_ratio": app.get("financial_ratio", 1.5),
            "ties_strength": app.get("ties_strength", 0.7),
            "days_to_travel": app.get("days_to_travel", 45),
            "prior_rejection": app.get("prior_rejection", 0),
            "visa_type_us_h1b": 1 if app.get("visa_type") == "us_h1b" else 0,
        }
        return np.array([[row[c] for c in self.feature_cols]]), row

    def score(self, app: ApplicationInput, top_k: int = 3) -> RiskResult:
        X, row = self._vectorize(app)
        prob = float(self.model.predict_proba(X)[0, 1])

        shap_values = self.explainer.shap_values(X)[0]
        contributions = sorted(
            zip(self.feature_cols, shap_values),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:top_k]

        top_factors = []
        for feat, contrib in contributions:
            direction = "increases" if contrib > 0 else "decreases"
            top_factors.append({
                "factor": FEATURE_LABELS.get(feat, feat),
                "value": row[feat],
                "direction": direction,
                "impact": round(float(contrib), 4),
            })

        if prob < 0.2:
            level = "low"
        elif prob < 0.5:
            level = "moderate"
        else:
            level = "high"

        return {
            "rejection_probability": round(prob, 4),
            "risk_level": level,
            "top_factors": top_factors,
        }


_model_singleton = None


def get_risk_model() -> RiskModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = RiskModel()
    return _model_singleton
