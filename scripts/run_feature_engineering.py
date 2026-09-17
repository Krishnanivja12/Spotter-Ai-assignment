
from __future__ import annotations
from pathlib import Path
import pandas as pd
from preprocessing import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    FreightDataCleaner,
    FreightFeatureEngineer,
    build_feature_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train-test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "feature_engineering"
TARGET = "posted_rate"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    train_x = train.drop(columns=[TARGET])

    # This first view shows the human-readable engineered columns before encoding.
    clean = FreightDataCleaner()
    engineer = FreightFeatureEngineer()
    engineered_train = engineer.fit_transform(clean.fit_transform(train_x))
    engineered_validation = engineer.transform(clean.transform(validation))
    engineered_train.head().to_csv(OUTPUT_DIR / "engineered_feature_example.csv", index=False)

    # Fit only on development features; validation is transformed without being fitted.
    pipeline = build_feature_pipeline()
    train_matrix = pipeline.fit_transform(train_x)
    validation_matrix = pipeline.transform(validation)
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    pd.DataFrame(train_matrix[:5].toarray() if hasattr(train_matrix, "toarray") else train_matrix[:5], columns=feature_names).to_csv(
        OUTPUT_DIR / "encoded_feature_example.csv", index=False
    )

    feature_catalog = pd.DataFrame([
        (column, "numeric", "raw") for column in MODEL_NUMERIC_COLUMNS if column not in ENGINEERED_NUMERIC_COLUMNS
    ] + [
        (column, "numeric", "engineered") for column in ENGINEERED_NUMERIC_COLUMNS
    ] + [
        (column, "categorical", "raw") for column in MODEL_CATEGORICAL_COLUMNS if column not in ENGINEERED_CATEGORICAL_COLUMNS
    ] + [
        (column, "categorical", "engineered") for column in ENGINEERED_CATEGORICAL_COLUMNS
    ], columns=["feature", "type", "source"])
    feature_catalog.to_csv(OUTPUT_DIR / "feature_catalog.csv", index=False)

    # Verification: same feature output width and no target/ID in final model matrix.
    assert len(engineered_train) == len(train)
    assert len(engineered_validation) == len(validation)
    assert TARGET not in engineered_train.columns
    assert "load_id" not in engineered_train.columns
    assert list(engineered_train.columns) == list(engineered_validation.columns)
    assert train_matrix.shape[1] == validation_matrix.shape[1]
    assert train_matrix.shape[0] == len(train)
    assert validation_matrix.shape[0] == len(validation)

    report = f"""# Phase 4 — Feature Engineering Report

No model was trained. All transformations use features available in both development and final validation data.

## Final pre-encoding columns

Numeric:\n```\n{MODEL_NUMERIC_COLUMNS}\n```

Categorical:\n```\n{MODEL_CATEGORICAL_COLUMNS}\n```

## Pipeline verification

- Development transformed shape: {train_matrix.shape}
- Validation transformed shape: {validation_matrix.shape}
- `posted_rate` excluded from X: yes
- `load_id` excluded from X: yes
- Train-only fitted pipeline transformed validation successfully: yes

## Rejected features

- Route string (`pickup -> delivery`): rejected because training has 4,014 routes and 1,461 validation rows use routes unseen in training. Individual cities and coordinates generalize more safely.
- Weight log/weight-per-distance: rejected because Phase 2 showed weak marginal weight correlation and there is no evidence yet that extra transformations improve validation performance.
- Distance squared: rejected because `distance_log1p` provides one compact nonlinear transform without unnecessary scale expansion.
- Target/route mean encodings: rejected because they can leak target information unless calculated fold-by-fold.
"""
    (OUTPUT_DIR / "feature_engineering_report.md").write_text(report, encoding="utf-8")
    print(f"Created Phase 4 outputs in: {OUTPUT_DIR.relative_to(ROOT)}")
    print("Feature-pipeline verification passed; no model was trained and raw CSVs were read only.")


if __name__ == "__main__":
    main()
