from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "posted_rate"
DATE_COLUMN = "date"
CHRONOLOGICAL_CUTOFF = pd.Timestamp("2025-09-01")


def split_data(frame: pd.DataFrame, cutoff: pd.Timestamp = CHRONOLOGICAL_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    dated = frame.copy()
    dated[DATE_COLUMN] = pd.to_datetime(dated[DATE_COLUMN], errors="raise")
    train = dated.loc[dated[DATE_COLUMN] < cutoff].copy()
    holdout = dated.loc[dated[DATE_COLUMN] >= cutoff].copy()
    if train.empty or holdout.empty:
        raise ValueError("Cutoff must leave non-empty train and holdout partitions.")
    return train, holdout
