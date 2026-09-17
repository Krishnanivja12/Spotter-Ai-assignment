
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "final_ridge_pipeline.joblib"
DECEMBER_PATH = ROOT / "data" / "december-chart-inputs.csv"
REQUIRED_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
EXPECTED_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")


def verify_december(frame: pd.DataFrame) -> None:
    """Verify the scorer-required December schema and fixed inputs locally."""
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Expected December columns in this order: {REQUIRED_COLUMNS}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if len(frame) != 31 or dates.isna().any() or dates.duplicated().any() or set(dates) != set(EXPECTED_DATES):
        raise ValueError("December file must contain each 2025-12-01 through 2025-12-31 date once")
    if not frame["pickup"].eq("Lexington").all() or not frame["delivery"].eq("Fort Wayne").all():
        raise ValueError("December route does not match the required fixed route")
    if not np.isclose(pd.to_numeric(frame["distance"], errors="coerce"), 360.0).all():
        raise ValueError("December distance must be 360")
    if not frame["equipment"].eq("Dry Van").all():
        raise ValueError("December equipment must be Dry Van")
    if not np.isclose(pd.to_numeric(frame["weight"], errors="coerce"), 32000.0).all():
        raise ValueError("December weight must be 32000")
    predicted = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if predicted.isna().any() or not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("December predicted_rate must be finite and strictly positive")


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Final model artifact not found: {MODEL_PATH}")
    december = pd.read_csv(DECEMBER_PATH)
    if list(december.columns) != REQUIRED_COLUMNS:
        raise ValueError("Unexpected December input schema; refusing to overwrite it")
    model = joblib.load(MODEL_PATH)
    features = december.drop(columns=["predicted_rate"])
    december["predicted_rate"] = model.predict(features)
    verify_december(december)
    december.to_csv(DECEMBER_PATH, index=False)
    print(f"Completed and verified 31 December predictions: {DECEMBER_PATH.relative_to(ROOT)}")
    print(december.head().to_string(index=False))


if __name__ == "__main__":
    main()
