"""Phase 6: train and evaluate a simple Ridge regression baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from preprocessing import (
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    FreightDataCleaner,
    FreightFeatureEngineer,
    build_model_preprocessor,
)
from splits import split_data


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train-test.csv"
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "baseline"
TARGET = "posted_rate"
RIDGE_ALPHA = 10.0


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("clean", FreightDataCleaner()),
            ("features", FreightFeatureEngineer()),
            ("preprocess", build_model_preprocessor(MODEL_NUMERIC_COLUMNS, MODEL_CATEGORICAL_COLUMNS, scale_numeric=True)),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    fit_frame, holdout_frame = split_data(data)
    x_fit, y_fit = fit_frame.drop(columns=[TARGET]), fit_frame[TARGET]
    x_holdout, y_holdout = holdout_frame.drop(columns=[TARGET]), holdout_frame[TARGET]

    baseline = build_pipeline()
    baseline.fit(x_fit, y_fit)
    predictions = baseline.predict(x_holdout)
    metrics = {
        "model": "Ridge regression baseline",
        "ridge_alpha": RIDGE_ALPHA,
        "fit_rows": int(len(x_fit)),
        "holdout_rows": int(len(x_holdout)),
        "mae": float(mean_absolute_error(y_holdout, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_holdout, predictions))),
        "r2": float(r2_score(y_holdout, predictions)),
        "minimum_prediction": float(predictions.min()),
        "maximum_prediction": float(predictions.max()),
        "non_positive_prediction_count": int((predictions <= 0).sum()),
    }
    (OUTPUT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    results = holdout_frame[["load_id", "date", "pickup", "delivery", "equipment", "distance", TARGET]].copy()
    results["predicted_rate"] = predictions
    results["signed_error"] = results["predicted_rate"] - results[TARGET]
    results["absolute_error"] = results["signed_error"].abs()
    results.to_csv(OUTPUT_DIR / "baseline_holdout_predictions.csv", index=False)

    equipment_errors = results.groupby("equipment").agg(
        rows=("load_id", "size"), mae=("absolute_error", "mean"), mean_signed_error=("signed_error", "mean")
    )
    equipment_errors.to_csv(OUTPUT_DIR / "baseline_errors_by_equipment.csv")

    distance_bins = pd.cut(results["distance"], bins=[0, 500, 1000, 2000, np.inf], labels=["0-500", "500-1000", "1000-2000", "2000+"])
    distance_errors = results.groupby(distance_bins, observed=False).agg(
        rows=("load_id", "size"), mae=("absolute_error", "mean"), mean_signed_error=("signed_error", "mean")
    )
    distance_errors.to_csv(OUTPUT_DIR / "baseline_errors_by_distance_bin.csv")

    target_deciles = pd.qcut(results[TARGET], q=10, duplicates="drop")
    target_errors = results.groupby(target_deciles, observed=False).agg(
        rows=("load_id", "size"), actual_mean=(TARGET, "mean"), predicted_mean=("predicted_rate", "mean"),
        mae=("absolute_error", "mean"), mean_signed_error=("signed_error", "mean"),
    )
    target_errors.to_csv(OUTPUT_DIR / "baseline_errors_by_actual_rate_decile.csv")
    results.sample(n=10, random_state=42).to_csv(OUTPUT_DIR / "baseline_prediction_examples.csv", index=False)
    results.nlargest(20, "absolute_error").to_csv(OUTPUT_DIR / "baseline_largest_errors.csv", index=False)

    # Verify no future rows were used for fitting, and every holdout row received a finite prediction.
    assert fit_frame["date"].max() < holdout_frame["date"].min()
    assert len(predictions) == len(holdout_frame)
    assert np.isfinite(predictions).all()
    assert results["load_id"].is_unique

    report = f"""# Phase 6 — Baseline Regression Results

## Baseline definition

The baseline is Ridge regression (`alpha={RIDGE_ALPHA}`): a regularized linear model. It uses the shared cleaning and feature pipeline, including training-only numeric imputation, unknown-category-safe one-hot encoding, and standardization for numeric features. It is fitted only on the January–August partition and evaluated on the later September–October holdout.

## Internal holdout metrics

| Metric | Value |
|---|---:|
| MAE | {metrics['mae']:.2f} |
| RMSE | {metrics['rmse']:.2f} |
| R² | {metrics['r2']:.4f} |
| Minimum prediction | {metrics['minimum_prediction']:.2f} |
| Maximum prediction | {metrics['maximum_prediction']:.2f} |
| Non-positive predictions | {metrics['non_positive_prediction_count']} |

These are internal chronological-holdout metrics, not the official Spotter score.
The raw baseline produces some non-positive estimates, so it is not suitable as a final submission model without a documented positive-prediction strategy.

## Error analysis

By equipment:\n```\n{equipment_errors.to_string()}\n```

By distance bin:\n```\n{distance_errors.to_string()}\n```

By actual-rate decile:\n```\n{target_errors.to_string()}\n```

The largest individual errors are saved in `baseline_largest_errors.csv`; reproducible examples are in `baseline_prediction_examples.csv`.
"""
    (OUTPUT_DIR / "baseline_report.md").write_text(report, encoding="utf-8")
    print(f"Created baseline results in: {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"Internal holdout MAE: {metrics['mae']:.2f}; RMSE: {metrics['rmse']:.2f}; R2: {metrics['r2']:.4f}")
    print("No final model was trained.")


if __name__ == "__main__":
    main()
