
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import build_final_ridge_pipeline


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train-test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
DECEMBER_PATH = ROOT / "data" / "december-chart-inputs.csv"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs" / "modeling"
TARGET = "posted_rate"
RIDGE_ALPHA = 10.0
PREDICTION_FLOOR = 0.01


def prediction_summary(predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(predictions)),
        "minimum": float(predictions.min()),
        "maximum": float(predictions.max()),
        "non_finite": int((~np.isfinite(predictions)).sum()),
        "non_positive": int((predictions <= 0).sum()),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    december = pd.read_csv(DECEMBER_PATH)
    x_development = development.drop(columns=[TARGET])
    y_development = development[TARGET]

    final_pipeline = build_final_ridge_pipeline(alpha=RIDGE_ALPHA, prediction_floor=PREDICTION_FLOOR)
    final_pipeline.fit(x_development, y_development)

    artifact_path = ARTIFACT_DIR / "final_ridge_pipeline.joblib"
    joblib.dump(final_pipeline, artifact_path)
    loaded_pipeline = joblib.load(artifact_path)

    validation_predictions = loaded_pipeline.predict(validation)
    december_features = december.drop(columns=["predicted_rate"], errors="ignore")
    december_predictions = loaded_pipeline.predict(december_features)
    assert len(validation_predictions) == len(validation)
    assert len(december_predictions) == len(december)
    assert np.isfinite(validation_predictions).all() and np.isfinite(december_predictions).all()
    assert (validation_predictions > 0).all() and (december_predictions > 0).all()
    assert np.allclose(validation_predictions, final_pipeline.predict(validation))

    manifest = {
        "model": "PositiveRidgeRegressor",
        "ridge_alpha": RIDGE_ALPHA,
        "prediction_floor": PREDICTION_FLOOR,
        "target": TARGET,
        "training_rows": int(len(development)),
        "training_date_min": str(pd.to_datetime(development["date"]).min().date()),
        "training_date_max": str(pd.to_datetime(development["date"]).max().date()),
        "artifact": str(artifact_path.relative_to(ROOT)),
        "validation_inference": prediction_summary(validation_predictions),
        "december_inference": prediction_summary(december_predictions),
        "december_adapter_policy": (
            "Fill missing city coordinates from development-feature city medians; leave unavailable market_index "
            "and quote_signal as missing for the pipeline imputer."
        ),
    }
    (OUTPUT_DIR / "final_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved final pipeline: {artifact_path.relative_to(ROOT)}")
    print("Final pipeline verification passed for validation and December input schemas.")


if __name__ == "__main__":
    main()
