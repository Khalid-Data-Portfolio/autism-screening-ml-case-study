from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_autism_screening_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    sample = {
        "a1_score": 1,
        "a2_score": 1,
        "a3_score": 0,
        "a4_score": 1,
        "a5_score": 1,
        "a6_score": 0,
        "a7_score": 1,
        "a8_score": 1,
        "a9_score": 0,
        "a10_score": 1,
        "age": 28,
        "gender": "m",
        "ethnicity": "White-European",
        "jaundice": "no",
        "family_asd": "yes",
        "country_of_residence": "United States",
        "used_app_before": "no",
        "relation": "Self",
    }

    X = pd.DataFrame([sample], columns=metadata["features"])
    probability = model.predict_proba(X)[0, 1]
    prediction = model.predict(X)[0]

    print("Sample autism screening prediction")
    print(f"Predicted class: {'ASD screening positive' if prediction == 1 else 'No ASD screening positive'}")
    print(f"Estimated probability: {probability:.3f}")
    print("Note: This is a screening model for portfolio/demo use, not a medical diagnosis.")


if __name__ == "__main__":
    main()
