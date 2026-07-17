"""
generate_synthetic_data.py — Builds a synthetic visa-application dataset with rejection
outcomes, grounded in the rejection patterns documented in data/policies/*.md
(late applications, insufficient funds, passport validity, missing ties-to-home-country
proof, etc). Real embassy data isn't public, so we simulate realistic feature
distributions and a rejection rule (+ noise) instead of hand-labeling.

Usage:
    python generate_synthetic_data.py
Outputs:
    data/synthetic_applications.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).parent / "data" / "synthetic_applications.csv"
N_SAMPLES = 6000
SEED = 42

VISA_TYPES = ["us_h1b", "schengen_tourist"]


def simulate(n=N_SAMPLES, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    visa_type = rng.choice(VISA_TYPES, size=n)

    # --- Document completeness: fraction of required docs present (0-1) ---
    doc_completeness = np.clip(rng.beta(5, 2, size=n), 0, 1)

    # --- Passport validity buffer in days beyond the minimum required ---
    # negative = below minimum (a hard rejection trigger)
    passport_buffer_days = rng.normal(120, 100, size=n).round().astype(int)

    # --- Financial proof strength: ratio of available funds to required minimum ---
    financial_ratio = np.clip(rng.lognormal(mean=0.15, sigma=0.5, size=n), 0.1, 5.0)

    # --- Employment / ties-to-home-country strength, 0-1 proxy score ---
    ties_strength = np.clip(rng.beta(4, 2.5, size=n), 0, 1)

    # --- Days between application submission and intended travel/start date ---
    # too early or too late both raise risk; we encode "days_to_travel"
    days_to_travel = rng.normal(45, 30, size=n).round().astype(int)
    days_to_travel = np.clip(days_to_travel, -10, 200)

    # --- Prior visa rejection history (binary) ---
    prior_rejection = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    # --- Underlying rejection probability (logistic combination of risk factors) ---
    # Coefficients are illustrative, reflecting the rejection reasons in the policy docs:
    # missing docs, thin passport buffer, weak funds, weak ties, bad timing, prior rejection.
    z = (
        0.75
        + (-2.5) * (1 - doc_completeness)                     # incomplete docs -> risk
        + (-0.006) * np.clip(passport_buffer_days, -400, 400)  # thin/negative buffer -> risk
        + (-1.0) * (financial_ratio - 1.0)                    # weak funds -> risk
        + (-1.5) * (1 - ties_strength)                        # weak ties -> risk
        + 0.03 * np.clip(-days_to_travel, 0, 60)               # applied too close to travel
        + 1.2 * prior_rejection                                # prior rejection -> risk
    )
    prob_reject = 1 / (1 + np.exp(-z))
    rejected = rng.binomial(1, prob_reject)

    df = pd.DataFrame({
        "visa_type": visa_type,
        "doc_completeness": doc_completeness.round(3),
        "passport_buffer_days": passport_buffer_days,
        "financial_ratio": financial_ratio.round(3),
        "ties_strength": ties_strength.round(3),
        "days_to_travel": days_to_travel,
        "prior_rejection": prior_rejection,
        "rejected": rejected,
    })
    return df


if __name__ == "__main__":
    df = simulate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(f"Rejection rate: {df['rejected'].mean():.2%}")
