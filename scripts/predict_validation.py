
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from verify_validation_predictions import check_predictions


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "final_ridge_pipeline.joblib"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
TEMPLATE_PATH = ROOT / "data" / "validation_predictions_template.csv"
OUTPUT_PATH = ROOT / "validation_predictions.csv"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Final model artifact not found: {MODEL_PATH}")
    validation = pd.read_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("The supplied template must contain load_id,predicted_rate")
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("Validation or template has duplicate load IDs")
    if set(validation["load_id"].astype(str)) != set(template["load_id"].astype(str)):
        raise ValueError("Validation and template load ID sets do not match")

    model = joblib.load(MODEL_PATH)
    validation_predictions = model.predict(validation)
    if len(validation_predictions) != len(validation) or not np.isfinite(validation_predictions).all():
        raise ValueError("Model did not return one finite prediction per validation row")

    prediction_by_id = pd.Series(validation_predictions, index=validation["load_id"].astype(str))
    output = template[["load_id"]].copy()
    output["predicted_rate"] = output["load_id"].astype(str).map(prediction_by_id)
    output.to_csv(OUTPUT_PATH, index=False)
    check_predictions(OUTPUT_PATH, expected_order=template["load_id"])
    print(f"Created and verified: {OUTPUT_PATH.relative_to(ROOT)}")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
