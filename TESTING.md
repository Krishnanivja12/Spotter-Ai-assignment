# Testing

This project was tested using the actual repository scripts and data files in this workspace.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

```bash
python scripts/train_final_model.py
```

## Evaluation

```bash
python scripts/run_split_strategy.py
python scripts/run_model_experiments.py
python scripts/run_baseline.py
```

## Generate validation predictions

```bash
python scripts/predict_validation.py
```

## Generate December predictions

```bash
python scripts/predict_december.py
```

## Run official scorer

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

## Verify outputs

```bash
python -m compileall .
```

```bash
python scripts/verify_validation_predictions.py --predictions data/validation_predictions.csv --template data/validation_predictions_template.csv
```

## Expected output

After the commands succeed:

- `artifacts/final_ridge_pipeline.joblib` is created.
- `validation_predictions.csv` contains exactly 12,000 rows and 2 columns.
- `data/december-chart-inputs.csv` contains 31 rows with dates from 2025-12-01 through 2025-12-31.
- `scorer_results/candidate_december.png` is generated.
- The scorer prints a successful validation summary.
