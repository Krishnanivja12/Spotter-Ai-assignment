# Phase 7 — Model Experimentation and Comparison

All candidates use the identical Phase 5 chronological split: January–August fitting data and September–October holdout data. Cleaning, feature engineering, imputation, and encoding are fitted only on the fitting partition. These are internal evaluation metrics, not the official Spotter score.

## Comparison table

```
                 model                       description  training_seconds    fit_mae   fit_rmse   fit_r2  holdout_mae  holdout_rmse  holdout_r2  rmse_generalization_gap  non_positive_holdout_predictions
                 ridge      Regularized linear benchmark          0.366918 133.233272 585.097842 0.842964   149.632525    643.533560    0.822171                58.435718                                11
           extra_trees Highly randomized nonlinear trees         71.502879  55.740560 334.387150 0.948709   156.297496    661.283875    0.812225               326.896725                                 0
         random_forest            Bagged nonlinear trees         36.351292  71.585471 386.488880 0.931480   160.268553    655.923499    0.815257               269.434619                                 0
hist_gradient_boosting           Boosted nonlinear trees          6.679601 111.146575 459.950812 0.902957   166.608031    652.895966    0.816959               192.945153                                 0
```

## Evidence-based selection

The leading candidate by chronological-holdout MAE is **ridge** with MAE 149.63, RMSE 643.53, and R² 0.8222. This identifies the candidate to carry into final-model work; it is not a final submission model yet.

## Overfitting check

Fit versus holdout metrics and `rmse_generalization_gap` are included in the table. A large positive gap means the candidate fits historical rows substantially better than later unseen rows and therefore requires caution.
