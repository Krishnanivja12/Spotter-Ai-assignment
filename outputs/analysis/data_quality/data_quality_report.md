# Phase 3 — Data Quality Report

All counts below are calculated from raw input files. No raw CSV has been changed.

## Missing values

Train:
```
load_id           0
pickup            0
delivery          0
pickup_lat        0
pickup_lon        0
delivery_lat      0
delivery_lon      0
distance          0
equipment         0
weight          300
date              0
market_index    374
quote_signal      0
posted_rate       0
```
Validation:
```
load_id           0
pickup            0
delivery          0
pickup_lat        0
pickup_lon        0
delivery_lat      0
delivery_lon      0
distance          0
equipment         0
weight          165
date              0
market_index    249
quote_signal      0
```

## Duplicate and identifier checks

Train:
```
                             count
row_count                    48000
exact_duplicate_rows             0
duplicate_load_ids               0
missing_load_ids                 0
invalid_dates_after_parsing      0
negative_weights               292
zero_weights                     0
non_positive_posted_rates        0
```
Validation:
```
                             count
row_count                    12000
exact_duplicate_rows             0
duplicate_load_ids               0
missing_load_ids                 0
invalid_dates_after_parsing      0
negative_weights               145
zero_weights                     0
```

## Numeric validity checks

Train:
```
                             rule  missing_count  non_finite_count  rule_violation_count      minimum      maximum
column                                                                                                            
pickup_lat      outside [-90, 90]              0                 0                     0     28.35765     44.30296
pickup_lon    outside [-180, 180]              0                 0                     0   -121.69849    -69.50000
delivery_lat    outside [-90, 90]              0                 0                     0     28.35765     44.30296
delivery_lon  outside [-180, 180]              0                 0                     0   -121.69849    -69.50000
distance             non-positive              0                 0                     0     70.00000   3439.80000
weight                   negative            300                 0                   292 -47500.00000  47500.00000
market_index         non-positive            374                 0                     0      0.67639      1.46778
quote_signal         non-positive              0                 0                     0      0.69228      3.61035
posted_rate          non-positive              0                 0                     0     57.22000  25533.00000
```
Validation:
```
                             rule  missing_count  non_finite_count  rule_violation_count      minimum      maximum
column                                                                                                            
pickup_lat      outside [-90, 90]              0                 0                     0     25.50000     44.30296
pickup_lon    outside [-180, 180]              0                 0                     0   -121.69849    -69.50000
delivery_lat    outside [-90, 90]              0                 0                     0     25.50000     44.30296
delivery_lon  outside [-180, 180]              0                 0                     0   -121.69849    -69.50000
distance             non-positive              0                 0                     0     70.00000   3325.70000
weight                   negative            165                 0                   145 -47500.00000  47500.00000
market_index         non-positive            249                 0                     0      0.72361      1.09895
quote_signal         non-positive              0                 0                     0      1.23476      2.89635
```

## Category comparison

```
           train_unique  validation_unique                                                         validation_only_values
pickup               64                 72  Allentown; Charlotte; Chicago; Jackson; Knoxville; Laredo; Norfolk; San Diego
delivery             64                 72  Allentown; Charlotte; Chicago; Jackson; Knoxville; Laredo; Norfolk; San Diego
equipment             3                  3                                                                               
```

## Before/after deterministic cleaning

Negative weights are converted to missing values, and `weight_was_negative` retains their origin. No rows are removed and no imputation is fitted here.

```
 train_raw_missing_weight  train_negative_weight  train_cleaned_missing_weight  validation_raw_missing_weight  validation_negative_weight  validation_cleaned_missing_weight  train_negative_weight_flag_sum  validation_negative_weight_flag_sum
                      300                    292                           592                            165                         145                                310                             292                                  145
```

## Leakage audit

- `posted_rate` exists only in training and is excluded before cleaning features.
- `load_id` is an identifier and must be excluded from model features.
- Train and validation have different, chronological date ranges; future validation rows are not used to fit any imputer or encoder.
- The reusable sklearn preprocessor was fitted on development features only in this verification; it transformed validation, including unseen cities, without error.
- `market_index` and `quote_signal` are present in final validation, so they are available-at-inference candidate features. Their business provenance should be confirmed before modeling; this structural audit found no direct target column leakage.
