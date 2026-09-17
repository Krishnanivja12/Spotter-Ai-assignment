# Phase 4 — Feature Engineering Report

No model was trained. All transformations use features available in both development and final validation data.

## Final pre-encoding columns

Numeric:
```
['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'distance', 'weight', 'market_index', 'quote_signal', 'weight_was_negative', 'distance_log1p', 'latitude_difference', 'longitude_difference', 'haversine_distance', 'distance_x_market_index', 'day_of_year_sin', 'day_of_year_cos']
```

Categorical:
```
['pickup', 'delivery', 'equipment', 'day_of_week']
```

## Pipeline verification

- Development transformed shape: (48000, 157)
- Validation transformed shape: (12000, 157)
- `posted_rate` excluded from X: yes
- `load_id` excluded from X: yes
- Train-only fitted pipeline transformed validation successfully: yes

## Rejected features

- Route string (`pickup -> delivery`): rejected because training has 4,014 routes and 1,461 validation rows use routes unseen in training. Individual cities and coordinates generalize more safely.
- Weight log/weight-per-distance: rejected because Phase 2 showed weak marginal weight correlation and there is no evidence yet that extra transformations improve validation performance.
- Distance squared: rejected because `distance_log1p` provides one compact nonlinear transform without unnecessary scale expansion.
- Target/route mean encodings: rejected because they can leak target information unless calculated fold-by-fold.
