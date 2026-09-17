
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from splits import CHRONOLOGICAL_CUTOFF, split_data


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train-test.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "split_strategy"
TARGET = "posted_rate"
RANDOM_STATE = 42


def partition_summary(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "partition": name,
        "rows": len(frame),
        "share_pct": 100 * len(frame) / 48_000,
        "first_date": frame["date"].min().date(),
        "last_date": frame["date"].max().date(),
        "unique_dates": frame["date"].nunique(),
        "mean_posted_rate": frame[TARGET].mean(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TRAIN_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    chronological_train, chronological_holdout = split_data(data)
    random_train, random_holdout = train_test_split(data, test_size=0.20, random_state=RANDOM_STATE, shuffle=True)

    monthly_counts = data.groupby(data["date"].dt.to_period("M")).size().rename("row_count")
    monthly_counts.to_csv(OUTPUT_DIR / "development_rows_by_month.csv")
    split_summary = pd.DataFrame([
        partition_summary("chronological_train", chronological_train),
        partition_summary("chronological_holdout", chronological_holdout),
        partition_summary("random_train_seed_42", random_train),
        partition_summary("random_holdout_seed_42", random_holdout),
    ])
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    # Assertions document the no-future-information boundary.
    assert chronological_train["date"].max() < CHRONOLOGICAL_CUTOFF
    assert chronological_holdout["date"].min() >= CHRONOLOGICAL_CUTOFF
    assert set(chronological_train["load_id"]).isdisjoint(chronological_holdout["load_id"])
    assert len(chronological_train) + len(chronological_holdout) == len(data)
    assert set(chronological_train["date"]).isdisjoint(set(chronological_holdout["date"]))

    report = f"""# Phase 5 — Train/Validation Split Strategy

## Actual development date distribution

- Development date range: {data['date'].min().date()} to {data['date'].max().date()}
- Final unlabeled validation date range (from Phase 2): 2025-11-01 to 2025-12-31
- Rows per development month:\n```\n{monthly_counts.to_string()}\n```

## Selected chronological split

- Cutoff: `{CHRONOLOGICAL_CUTOFF.date()}`
- Fit partition: dates before cutoff (2025-01-01 through {chronological_train['date'].max().date()}), {len(chronological_train):,} rows
- Internal holdout: cutoff and later (2025-09-01 through {chronological_holdout['date'].max().date()}), {len(chronological_holdout):,} rows
- The holdout covers {chronological_holdout['date'].nunique()} calendar days, matching the 61-day November–December final horizon.

## Random-split comparison

- Random seed: {RANDOM_STATE}; 80/20 split gives {len(random_train):,} fitting rows and {len(random_holdout):,} holdout rows.
- Both random partitions span {random_train['date'].min().date()} to {random_train['date'].max().date()} and share {len(set(random_train['date']) & set(random_holdout['date']))} dates.
- This makes random validation useful only as a secondary diagnostic, not the primary selection metric, because it lets a model learn from later months while evaluating earlier-month rows.

## Evaluation protocol

For every candidate model in later phases: split by date first; fit cleaning, feature engineering, encoding, and model only on the pre-2025-09-01 partition; predict the September–October holdout; calculate internal regression metrics there. Do not call these metrics the official Spotter score.
"""
    (OUTPUT_DIR / "split_strategy_report.md").write_text(report, encoding="utf-8")
    print(f"Created split strategy outputs in: {OUTPUT_DIR.relative_to(ROOT)}")
    print("Chronological split assertions passed. No model was trained.")


if __name__ == "__main__":
    main()
