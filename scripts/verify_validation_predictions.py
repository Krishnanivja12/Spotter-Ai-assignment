from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}


def check_predictions(path: Path, expected_order: pd.Series | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Expected exactly these columns in order: load_id,predicted_rate")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows; found {len(frame):,}")
    if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
        raise ValueError("load_id contains missing or duplicate values")
    if set(frame["load_id"].astype(str)) != EXPECTED_IDS:
        raise ValueError("load_id values do not exactly match TE-000001 through TE-012000")
    predictions = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if predictions.isna().any() or not np.isfinite(predictions).all():
        raise ValueError("predicted_rate contains missing, non-numeric, or non-finite values")
    if (predictions <= 0).any():
        raise ValueError("predicted_rate must be strictly positive")
    if expected_order is not None and not frame["load_id"].astype(str).reset_index(drop=True).equals(
        expected_order.astype(str).reset_index(drop=True)
    ):
        raise ValueError("Prediction order differs from the supplied template order")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="validation_predictions.csv")
    parser.add_argument("--template", default="data/validation_predictions_template.csv")
    args = parser.parse_args()
    template = pd.read_csv(args.template)
    verified = check_predictions(Path(args.predictions), expected_order=template["load_id"])
    print(f"Verified {len(verified):,} predictions with exact schema, IDs, template order, and positive numeric rates.")


if __name__ == "__main__":
    main()
