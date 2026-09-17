
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
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
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "experiments"
PREDICTIONS_DIR = OUTPUT_DIR / "holdout_predictions"
TARGET = "posted_rate"
RANDOM_STATE = 42


def build_pipeline(model: object, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        steps=[
            ("clean", FreightDataCleaner()),
            ("features", FreightFeatureEngineer()),
            (
                "preprocess",
                build_model_preprocessor(
                    MODEL_NUMERIC_COLUMNS,
                    MODEL_CATEGORICAL_COLUMNS,
                    scale_numeric=scale_numeric,
                    dense_output=True,
                ),
            ),
            ("model", model),
        ]
    )


def regression_metrics(actual: pd.Series, predicted: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": float(mean_absolute_error(actual, predicted)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        f"{prefix}_r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    fit_frame, holdout_frame = split_data(data)
    x_fit, y_fit = fit_frame.drop(columns=[TARGET]), fit_frame[TARGET]
    x_holdout, y_holdout = holdout_frame.drop(columns=[TARGET]), holdout_frame[TARGET]

    candidates = {
        "ridge": (Ridge(alpha=10.0), True, "Regularized linear benchmark"),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=250, max_features=0.8, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
            ),
            False,
            "Bagged nonlinear trees",
        ),
        "extra_trees": (
            ExtraTreesRegressor(
                n_estimators=250, max_features=0.8, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
            ),
            False,
            "Highly randomized nonlinear trees",
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0,
                early_stopping=False, random_state=RANDOM_STATE,
            ),
            False,
            "Boosted nonlinear trees",
        ),
    }

    comparison = []
    for name, (model, scale_numeric, description) in candidates.items():
        pipeline = build_pipeline(model, scale_numeric=scale_numeric)
        start = time.perf_counter()
        pipeline.fit(x_fit, y_fit)
        elapsed_seconds = time.perf_counter() - start
        fit_predictions = pipeline.predict(x_fit)
        holdout_predictions = pipeline.predict(x_holdout)
        row = {
            "model": name,
            "description": description,
            "training_seconds": elapsed_seconds,
            **regression_metrics(y_fit, fit_predictions, "fit"),
            **regression_metrics(y_holdout, holdout_predictions, "holdout"),
            "rmse_generalization_gap": float(
                np.sqrt(mean_squared_error(y_holdout, holdout_predictions)) - np.sqrt(mean_squared_error(y_fit, fit_predictions))
            ),
            "non_positive_holdout_predictions": int((holdout_predictions <= 0).sum()),
        }
        comparison.append(row)

        prediction_output = holdout_frame[["load_id", "date", TARGET]].copy()
        prediction_output["predicted_rate"] = holdout_predictions
        prediction_output["signed_error"] = prediction_output["predicted_rate"] - prediction_output[TARGET]
        prediction_output["absolute_error"] = prediction_output["signed_error"].abs()
        prediction_output.to_csv(PREDICTIONS_DIR / f"{name}_predictions.csv", index=False)

    table = pd.DataFrame(comparison).sort_values(["holdout_mae", "holdout_rmse"], ascending=True).reset_index(drop=True)
    table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    selected = table.iloc[0]
    report = f"""# Phase 7 — Model Experimentation and Comparison

All candidates use the identical Phase 5 chronological split: January–August fitting data and September–October holdout data. Cleaning, feature engineering, imputation, and encoding are fitted only on the fitting partition. These are internal evaluation metrics, not the official Spotter score.

## Comparison table

```\n{table.to_string(index=False)}\n```

## Evidence-based selection

The leading candidate by chronological-holdout MAE is **{selected['model']}** with MAE {selected['holdout_mae']:.2f}, RMSE {selected['holdout_rmse']:.2f}, and R² {selected['holdout_r2']:.4f}. This identifies the candidate to carry into final-model work; it is not a final submission model yet.

## Overfitting check

Fit versus holdout metrics and `rmse_generalization_gap` are included in the table. A large positive gap means the candidate fits historical rows substantially better than later unseen rows and therefore requires caution.
"""
    (OUTPUT_DIR / "model_experiment_report.md").write_text(report, encoding="utf-8")
    print(f"Created model-comparison outputs in: {OUTPUT_DIR.relative_to(ROOT)}")
    print(table[["model", "holdout_mae", "holdout_rmse", "holdout_r2", "training_seconds"]].to_string(index=False))
    print("No final model was trained.")


if __name__ == "__main__":
    main()
