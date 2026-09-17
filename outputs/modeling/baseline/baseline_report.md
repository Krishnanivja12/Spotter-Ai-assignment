# Phase 6 — Baseline Regression Results

## Baseline definition

The baseline is Ridge regression (`alpha=10.0`): a regularized linear model. It uses the shared cleaning and feature pipeline, including training-only numeric imputation, unknown-category-safe one-hot encoding, and standardization for numeric features. It is fitted only on the January–August partition and evaluated on the later September–October holdout.

## Internal holdout metrics

| Metric | Value |
|---|---:|
| MAE | 149.70 |
| RMSE | 643.54 |
| R² | 0.8222 |
| Minimum prediction | -181.80 |
| Maximum prediction | 6416.98 |
| Non-positive predictions | 11 |

These are internal chronological-holdout metrics, not the official Spotter score.
The raw baseline produces some non-positive estimates, so it is not suitable as a final submission model without a documented positive-prediction strategy.

## Error analysis

By equipment:
```
           rows         mae  mean_signed_error
equipment                                     
Dry Van    5360  135.582267         -84.874225
Flatbed    1770  141.617950         -71.599698
Reefer     2393  187.281607         -86.920844
```

By distance bin:
```
           rows         mae  mean_signed_error
distance                                      
0-500      2090  113.528763         -29.825037
500-1000   2972   93.912677         -36.133769
1000-2000  2927  173.709522        -128.327988
2000+      1534  261.224412        -159.269218
```

By actual-rate decile:
```
                              rows  actual_mean  predicted_mean         mae  mean_signed_error
posted_rate                                                                                   
(104.37899999999999, 808.37]   953   570.666464      604.731987  168.028122          34.065523
(808.37, 1106.968]             952   959.135651      967.409634   99.982771           8.273983
(1106.968, 1401.906]           952  1249.487300     1249.389882   76.042594          -0.097418
(1401.906, 1714.204]           952  1560.306408     1551.604730   62.474914          -8.701678
(1714.204, 2044.32]            953  1879.126107     1872.316661   65.436108          -6.809446
(2044.32, 2421.498]            952  2222.487962     2190.935270   59.748891         -31.552692
(2421.498, 2974.21]            952  2679.084601     2618.271662   80.520488         -60.812939
(2974.21, 3712.884]            952  3339.678214     3252.551787  112.391296         -87.126427
(3712.884, 4426.256]           952  4059.333487     3950.115389  151.037503        -109.218099
(4426.256, 20361.89]           953  5404.484439     4837.556890  620.865966        -566.927549
```

The largest individual errors are saved in `baseline_largest_errors.csv`; reproducible examples are in `baseline_prediction_examples.csv`.
