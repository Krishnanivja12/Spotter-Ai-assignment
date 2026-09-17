# Freight Rate Prediction for Spotter ML Assessment

This repository contains a reproducible freight-rate prediction workflow designed for the Freight Rate ML assessment. The objective is to learn a mapping from shipment attributes to a target `posted_rate` using the labeled development dataset, validate the model on a realistic future-like split, generate final predictions for the validation loads, and produce the required scorer-compatible December output.

## 1. Problem statement

The task is a supervised regression problem: predict the freight rate for each load based on route, equipment, weight, distance, date, and market/quote indicators.

The challenge requirements are to:

- train on the labeled development data in `data/train-test.csv`
- validate the model using a chronology-aware split
- predict all rows in `data/validation.csv`
- fill `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`
- predict the December scenario data in `data/december-chart-inputs.csv`
- run the project scorer with `score.py`
- produce a final report and a submission-ready GitHub repository

## 2. Dataset description

The raw input datasets are stored in the `data/` directory:

- `data/train-test.csv`: labeled development dataset with 48,000 rows and the target `posted_rate`
- `data/validation.csv`: 12,000 unlabeled validation loads with unique `load_id` values
- `data/validation_predictions_template.csv`: required template for final validation output
- `data/december-chart-inputs.csv`: 31 fixed-date scenario rows used to generate the December chart

Main columns include:

- `load_id`
- `pickup`, `delivery`
- `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`
- `distance`
- `equipment`
- `weight`
- `date`
- `market_index`
- `quote_signal`
- `posted_rate` (training target only)

## 3. Approach

The solution uses a reproducible sklearn-style pipeline with:

- deterministic data cleaning
- missing-value handling for numeric and categorical features
- feature engineering based on route geometry and date signals
- a chronological holdout strategy to emulate future prediction
- model comparison across candidate regressors
- a final Ridge-based pipeline selected using validation evidence

The model is trained on labeled historical data and used to generate predictions for both the final validation set and the December scenario.

## 4. Data-quality handling

The project treats data issues explicitly instead of silently converting or dropping them:

- missing numeric values are handled in the pipeline via imputation
- negative `weight` values are converted to missing and tracked with a flag feature
- categorical unknown values are handled safely with `handle_unknown="ignore"`
- training and prediction use the same preprocessing logic to prevent leakage
- no raw input files are overwritten

This prevents invalid values from being passed directly into the model while preserving the original dataset for auditability.

## 5. Feature engineering

The final model uses a compact feature set built from the raw data:

- distance-based features: `distance_log1p`, `distance_x_market_index`
- spatial features: `latitude_difference`, `longitude_difference`, `haversine_distance`
- date features: `day_of_week`, `day_of_year_sin`, `day_of_year_cos`
- validity features: `weight_was_negative`

The feature logic is centralized in `scripts/preprocessing.py` so the same workflow is reused during fit and inference.

## 6. Validation strategy

A chronology-aware split is used instead of a random split because the validation data represents later periods than the training data.

The project evaluates models on a holdout period that reflects a realistic forecasting scenario using future dates and preserves the temporal structure of the problem.

## 7. Model selection

The assessed candidate models include:

- Ridge regression
- random forest
- extra trees
- hist gradient boosting

The final selected model is a Ridge-based pipeline because it provides the best holdout performance while remaining interpretable, fast, and stable. The final pipeline also enforces a positive prediction floor required by the scorer.

## 8. How to install dependencies

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 9. How to train

```bash
python scripts/train_final_model.py
```

This script reads the labeled development data, fits the final pipeline, and saves the artifact to `artifacts/final_ridge_pipeline.joblib`.

## 10. How to generate predictions

Generate final validation predictions:

```bash
python scripts/predict_validation.py
```

This creates `validation_predictions.csv` using the saved model and the validation template.

Generate December scenario predictions:

```bash
python scripts/predict_december.py
```

This fills the `predicted_rate` column in `data/december-chart-inputs.csv` using the same preprocessing pipeline and saved model.

## 11. How to run score.py

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

The scorer validates the output files and creates the December chart in `scorer_results/candidate_december.png`.

## 12. Expected output files

After successful execution, the repository should contain the following outputs:

- `artifacts/final_ridge_pipeline.joblib`
- `validation_predictions.csv`
- `scorer_results/candidate_december.png`
- `reports/spotter_freight_rate_ml_assessment_report.docx`
- `outputs/analysis/project_inspection.txt`
- `outputs/analysis/eda/...` for exploratory analysis outputs
- `outputs/analysis/data_quality/...` for quality analysis outputs
- `outputs/analysis/feature_engineering/...` for feature engineering outputs
- `outputs/analysis/split_strategy/...` for split strategy outputs
- `outputs/modeling/baseline/...` for baseline outputs
- `outputs/modeling/experiments/...` for model comparison outputs
- `outputs/modeling/final_model_manifest.json`

## 13. Project structure

```text
.
├── artifacts/
│   └── final_ridge_pipeline.joblib
├── data/
│   ├── december-chart-inputs.csv
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── validation_predictions_template.csv
├── outputs/
│   ├── analysis/
│   │   ├── project_inspection.txt
│   │   ├── eda/
│   │   ├── data_quality/
│   │   ├── feature_engineering/
│   │   └── split_strategy/
│   └── modeling/
│       ├── baseline/
│       ├── experiments/
│       └── final_model_manifest.json
├── reports/
│   └── spotter_freight_rate_ml_assessment_report.docx
├── scripts/
│   ├── generate_assessment_report.py
│   ├── inspect_project.py
│   ├── predict_december.py
│   ├── predict_validation.py
│   ├── preprocessing.py
│   ├── run_baseline.py
│   ├── run_data_quality.py
│   ├── run_eda.py
│   ├── run_feature_engineering.py
│   ├── run_model_experiments.py
│   ├── run_split_strategy.py
│   ├── splits.py
│   └── train_final_model.py
├── .gitignore
├── freight-rate-ml-assessment.pdf
├── README.md
├── requirements.txt
├── score.py
├── validation_predictions.csv
└── scorer_results/
    └── candidate_december.png
```

## 14. Reproducibility instructions

To reproduce the project from a clean checkout:

```bash
git clone <repository-url>
cd <repository-name>
python -m pip install -r requirements.txt
python scripts/train_final_model.py
python scripts/predict_validation.py
python scripts/predict_december.py
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
python scripts/generate_assessment_report.py
```

Important reproducibility notes:

- do not edit raw CSV files in `data/`
- rely on the scripts in `scripts/` for all transformations
- keep the same feature pipeline during training and inference
- ensure Python dependencies are installed in the same environment used for generation

## 15. Limitations

- The project uses a single chronology-aware validation window rather than a broad multi-fold temporal validation study.
- Some extreme freight-rate events remain difficult to predict using a linear model.
- Temporal drift may affect model performance across future periods.
- The market-driven features should be interpreted with business context because they reflect market conditions, not only operational load attributes.

## 16. Repository review summary

This repository was reviewed for:

- broken imports: checked and the scripts use consistent local imports from the same project folder
- hardcoded local paths: avoided; paths are resolved from the file location using `Path(__file__).resolve()`
- missing dependencies: documented in `requirements.txt`
- unclear commands: standardised in this README with explicit execution steps
- unnecessary files: avoided or removed from the repository footprint where possible
- large temporary files: no large generated artifacts are being committed beyond the required deliverables
- accidental secrets: none included
- missing documentation: addressed by this README

## 17. Submission note

This repository is structured for a clean ML engineering submission and includes the source code, preprocessing logic, model training script, prediction scripts, score validation, and report generation workflow required by the assessment.
