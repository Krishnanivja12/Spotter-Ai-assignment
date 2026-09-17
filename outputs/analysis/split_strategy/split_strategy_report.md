# Phase 5 — Train/Validation Split Strategy

## Actual development date distribution

- Development date range: 2025-01-01 to 2025-10-31
- Final unlabeled validation date range (from Phase 2): 2025-11-01 to 2025-12-31
- Rows per development month:
```
date
2025-01    4918
2025-02    4337
2025-03    5036
2025-04    4819
2025-05    4913
2025-06    4783
2025-07    4912
2025-08    4759
2025-09    4670
2025-10    4853
Freq: M
```

## Selected chronological split

- Cutoff: `2025-09-01`
- Fit partition: dates before cutoff (2025-01-01 through 2025-08-31), 38,477 rows
- Internal holdout: cutoff and later (2025-09-01 through 2025-10-31), 9,523 rows
- The holdout covers 61 calendar days, matching the 61-day November–December final horizon.

## Random-split comparison

- Random seed: 42; 80/20 split gives 38,400 fitting rows and 9,600 holdout rows.
- Both random partitions span 2025-01-01 to 2025-10-31 and share 304 dates.
- This makes random validation useful only as a secondary diagnostic, not the primary selection metric, because it lets a model learn from later months while evaluating earlier-month rows.

## Evaluation protocol

For every candidate model in later phases: split by date first; fit cleaning, feature engineering, encoding, and model only on the pre-2025-09-01 partition; predict the September–October holdout; calculate internal regression metrics there. Do not call these metrics the official Spotter score.
