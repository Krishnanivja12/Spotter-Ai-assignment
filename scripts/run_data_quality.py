from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import FreightDataCleaner, build_model_preprocessor


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train-test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "data_quality"
TARGET = "posted_rate"
NUMERIC_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal",
]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]


def finite_invalid_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce")
    return int((values.notna() & ~np.isfinite(values)).sum())


def numeric_validity(frame: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    checks = []
    rules = {
        "pickup_lat": ("outside [-90, 90]", ~frame["pickup_lat"].between(-90, 90)),
        "pickup_lon": ("outside [-180, 180]", ~frame["pickup_lon"].between(-180, 180)),
        "delivery_lat": ("outside [-90, 90]", ~frame["delivery_lat"].between(-90, 90)),
        "delivery_lon": ("outside [-180, 180]", ~frame["delivery_lon"].between(-180, 180)),
        "distance": ("non-positive", frame["distance"].le(0)),
        "weight": ("negative", frame["weight"].lt(0)),
        "market_index": ("non-positive", frame["market_index"].le(0)),
        "quote_signal": ("non-positive", frame["quote_signal"].le(0)),
    }
    if has_target:
        rules[TARGET] = ("non-positive", frame[TARGET].le(0))
    for column, (rule, invalid) in rules.items():
        checks.append({
            "column": column,
            "rule": rule,
            "missing_count": int(frame[column].isna().sum()),
            "non_finite_count": finite_invalid_count(frame[column]),
            "rule_violation_count": int(invalid.fillna(False).sum()),
            "minimum": frame[column].min(skipna=True),
            "maximum": frame[column].max(skipna=True),
        })
    return pd.DataFrame(checks).set_index("column")


def quality_overview(frame: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    parsed_date = pd.to_datetime(frame["date"], errors="coerce")
    values = {
        "row_count": len(frame),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_load_ids": int(frame["load_id"].duplicated().sum()),
        "missing_load_ids": int(frame["load_id"].isna().sum()),
        "invalid_dates_after_parsing": int(parsed_date.isna().sum()),
        "negative_weights": int(frame["weight"].lt(0).sum()),
        "zero_weights": int(frame["weight"].eq(0).sum()),
    }
    if has_target:
        values["non_positive_posted_rates"] = int(frame[TARGET].le(0).sum())
    return pd.DataFrame.from_dict(values, orient="index", columns=["count"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    # Audit outputs from untouched, raw frames.
    quality_overview(train, has_target=True).to_csv(OUTPUT_DIR / "train_quality_overview.csv")
    quality_overview(validation, has_target=False).to_csv(OUTPUT_DIR / "validation_quality_overview.csv")
    train.isna().sum().rename("missing_count").to_csv(OUTPUT_DIR / "train_missing_values.csv")
    validation.isna().sum().rename("missing_count").to_csv(OUTPUT_DIR / "validation_missing_values.csv")
    numeric_validity(train, has_target=True).to_csv(OUTPUT_DIR / "train_numeric_validity.csv")
    numeric_validity(validation, has_target=False).to_csv(OUTPUT_DIR / "validation_numeric_validity.csv")

    category_comparison = pd.DataFrame({
        "train_unique": train[CATEGORICAL_COLUMNS].nunique(dropna=True),
        "validation_unique": validation[CATEGORICAL_COLUMNS].nunique(dropna=True),
        "validation_only_values": [
            "; ".join(map(str, sorted(set(validation[column].dropna()) - set(train[column].dropna()))))
            for column in CATEGORICAL_COLUMNS
        ],
    })
    category_comparison.to_csv(OUTPUT_DIR / "categorical_comparison.csv")

    cleaner = FreightDataCleaner()
    cleaned_train = cleaner.fit_transform(train.drop(columns=[TARGET]))
    cleaned_validation = cleaner.transform(validation)
    before_after = pd.DataFrame({
        "train_raw_missing_weight": [int(train["weight"].isna().sum())],
        "train_negative_weight": [int(train["weight"].lt(0).sum())],
        "train_cleaned_missing_weight": [int(cleaned_train["weight"].isna().sum())],
        "validation_raw_missing_weight": [int(validation["weight"].isna().sum())],
        "validation_negative_weight": [int(validation["weight"].lt(0).sum())],
        "validation_cleaned_missing_weight": [int(cleaned_validation["weight"].isna().sum())],
        "train_negative_weight_flag_sum": [int(cleaned_train["weight_was_negative"].sum())],
        "validation_negative_weight_flag_sum": [int(cleaned_validation["weight_was_negative"].sum())],
    })
    before_after.to_csv(OUTPUT_DIR / "cleaning_before_after.csv", index=False)

    # Tests demonstrate deterministic cleaning and schema consistency without fitting on validation.
    assert len(cleaned_train) == len(train)
    assert len(cleaned_validation) == len(validation)
    assert cleaned_train["weight"].lt(0).sum() == 0
    assert cleaned_validation["weight"].lt(0).sum() == 0
    assert cleaned_train["weight_was_negative"].sum() == train["weight"].lt(0).sum()
    assert cleaned_validation["weight_was_negative"].sum() == validation["weight"].lt(0).sum()
    assert list(cleaned_train.columns) == list(cleaned_validation.columns)

    # Demonstrate the future pipeline's unknown-category and train-only fit behavior.
    model_numeric = NUMERIC_COLUMNS + ["weight_was_negative"]
    model_preprocessor = build_model_preprocessor(model_numeric, CATEGORICAL_COLUMNS)
    train_matrix = model_preprocessor.fit_transform(cleaned_train[model_numeric + CATEGORICAL_COLUMNS])
    validation_matrix = model_preprocessor.transform(cleaned_validation[model_numeric + CATEGORICAL_COLUMNS])
    assert train_matrix.shape[0] == len(train)
    assert validation_matrix.shape[0] == len(validation)
    assert train_matrix.shape[1] == validation_matrix.shape[1]

    report = f"""# Phase 3 — Data Quality Report

All counts below are calculated from raw input files. No raw CSV has been changed.

## Missing values

Train:\n```\n{train.isna().sum().to_string()}\n```
Validation:\n```\n{validation.isna().sum().to_string()}\n```

## Duplicate and identifier checks

Train:\n```\n{quality_overview(train, True).to_string()}\n```
Validation:\n```\n{quality_overview(validation, False).to_string()}\n```

## Numeric validity checks

Train:\n```\n{numeric_validity(train, True).to_string()}\n```
Validation:\n```\n{numeric_validity(validation, False).to_string()}\n```

## Category comparison

```\n{category_comparison.to_string()}\n```

## Before/after deterministic cleaning

Negative weights are converted to missing values, and `weight_was_negative` retains their origin. No rows are removed and no imputation is fitted here.

```\n{before_after.to_string(index=False)}\n```

## Leakage audit

- `posted_rate` exists only in training and is excluded before cleaning features.
- `load_id` is an identifier and must be excluded from model features.
- Train and validation have different, chronological date ranges; future validation rows are not used to fit any imputer or encoder.
- The reusable sklearn preprocessor was fitted on development features only in this verification; it transformed validation, including unseen cities, without error.
- `market_index` and `quote_signal` are present in final validation, so they are available-at-inference candidate features. Their business provenance should be confirmed before modeling; this structural audit found no direct target column leakage.
"""
    (OUTPUT_DIR / "data_quality_report.md").write_text(report, encoding="utf-8")
    print(f"Created Phase 3 audit outputs in: {OUTPUT_DIR.relative_to(ROOT)}")
    print("Cleaning verification tests passed; raw CSV files were read only.")


if __name__ == "__main__":
    main()
