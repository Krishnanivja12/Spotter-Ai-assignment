from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train-test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis" / "eda"
PLOTS_DIR = OUTPUT_DIR / "plots"
TARGET = "posted_rate"
NUMERIC_FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal",
]
PRIMARY_COLOR = "#0B5C6B"
ACCENT_COLOR = "#D95F02"


def apply_plot_style() -> None:
    """Apply one clean, readable style to every EDA chart."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    })


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def format_series(series: pd.Series) -> str:
    return series.to_string()


def summary_stats(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def iqr_outlier_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = frame[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = ((values < lower) | (values > upper)).sum()
        rows.append({"feature": column, "lower_bound": lower, "upper_bound": upper,
                     "outlier_count": count, "outlier_pct": 100 * count / len(values)})
    return pd.DataFrame(rows).set_index("feature")


def write_section(lines: list[str], title: str, body: str) -> None:
    lines.extend([f"\n## {title}\n", body])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    validation["date"] = pd.to_datetime(validation["date"], errors="coerce")

    # Tabular outputs: easy to inspect or reuse in the later report.
    summary_stats(train, NUMERIC_FEATURES + [TARGET]).to_csv(OUTPUT_DIR / "train_numeric_summary.csv")
    summary_stats(validation, NUMERIC_FEATURES).to_csv(OUTPUT_DIR / "validation_numeric_summary.csv")
    train["equipment"].value_counts(dropna=False).rename("count").to_csv(OUTPUT_DIR / "train_equipment_counts.csv")
    validation["equipment"].value_counts(dropna=False).rename("count").to_csv(OUTPUT_DIR / "validation_equipment_counts.csv")
    train.groupby("equipment")[TARGET].agg(["count", "mean", "median", "std", "min", "max"]).to_csv(
        OUTPUT_DIR / "posted_rate_by_equipment.csv"
    )
    train.groupby(train["date"].dt.to_period("M"))[TARGET].agg(["count", "mean", "median", "std"]).to_csv(
        OUTPUT_DIR / "monthly_posted_rate.csv"
    )
    train[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False).rename("pearson_correlation").to_csv(
        OUTPUT_DIR / "numeric_target_correlations.csv"
    )
    iqr_outlier_summary(train, NUMERIC_FEATURES + [TARGET]).to_csv(OUTPUT_DIR / "iqr_outlier_summary.csv")
    distribution_shift = pd.DataFrame({
        "train_mean": train[NUMERIC_FEATURES].mean(),
        "validation_mean": validation[NUMERIC_FEATURES].mean(),
        "train_median": train[NUMERIC_FEATURES].median(),
        "validation_median": validation[NUMERIC_FEATURES].median(),
    })
    distribution_shift.to_csv(OUTPUT_DIR / "train_validation_numeric_comparison.csv")

    # Useful, compact plot set.
    plt.figure(figsize=(8, 4.5))
    plt.hist(train[TARGET], bins=60, color=PRIMARY_COLOR, edgecolor="white")
    plt.title("Training posted_rate distribution")
    plt.xlabel("posted_rate")
    plt.ylabel("Load count")
    save_plot("01_posted_rate_distribution.png")

    plt.figure(figsize=(8, 4.5))
    plt.hist(train["distance"], bins=60, color=PRIMARY_COLOR, edgecolor="white")
    plt.title("Training distance distribution")
    plt.xlabel("distance")
    plt.ylabel("Load count")
    save_plot("02_distance_distribution.png")

    scatter = train.sample(n=min(8_000, len(train)), random_state=42)
    plt.figure(figsize=(8, 5))
    plt.scatter(scatter["distance"], scatter[TARGET], s=6, alpha=0.18, color=PRIMARY_COLOR, label="Loads")
    slope, intercept = np.polyfit(train["distance"], train[TARGET], 1)
    x = np.linspace(train["distance"].min(), train["distance"].max(), 100)
    plt.plot(x, slope * x + intercept, color=ACCENT_COLOR, linewidth=2, label="Linear trend")
    plt.title("Distance vs posted_rate (8,000-row deterministic sample)")
    plt.xlabel("distance")
    plt.ylabel("posted_rate")
    plt.legend()
    save_plot("03_distance_vs_posted_rate.png")

    equipment_order = train.groupby("equipment")[TARGET].median().sort_values().index
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(
        [train.loc[train["equipment"].eq(equipment), TARGET].dropna() for equipment in equipment_order],
        tick_labels=equipment_order,
        showfliers=False,
    )
    plt.title("posted_rate by equipment (outliers hidden for readability)")
    plt.xlabel("equipment")
    plt.ylabel("posted_rate")
    save_plot("04_posted_rate_by_equipment.png")

    monthly = train.groupby(train["date"].dt.to_period("M"))[TARGET].mean()
    plt.figure(figsize=(8, 4.5))
    plt.plot(monthly.index.astype(str), monthly.values, marker="o", color=PRIMARY_COLOR, label="Monthly mean")
    plt.title("Monthly mean posted_rate in development data")
    plt.xlabel("month")
    plt.ylabel("mean posted_rate")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    save_plot("05_monthly_mean_posted_rate.png")

    top_pickups = train["pickup"].value_counts().head(15).sort_values()
    plt.figure(figsize=(8, 5.5))
    plt.barh(top_pickups.index, top_pickups.values, color=PRIMARY_COLOR)
    plt.title("Top 15 pickup cities in development data")
    plt.xlabel("Load count")
    save_plot("06_top_pickup_cities.png")

    plt.figure(figsize=(10, 7))
    for index, column in enumerate(["weight", "market_index", "quote_signal"], start=1):
        axis = plt.subplot(3, 1, index)
        axis.hist(train[column].dropna(), bins=50, density=True, alpha=0.55, label="Train", color=PRIMARY_COLOR)
        axis.hist(validation[column].dropna(), bins=50, density=True, alpha=0.45, label="Validation", color=ACCENT_COLOR)
        axis.set_title(f"{column}: train vs validation distribution")
        axis.set_ylabel("density")
        axis.legend()
    plt.xlabel("feature value")
    save_plot("07_train_validation_feature_distributions.png")

    # Human-readable report.
    validation_only = {
        column: sorted(set(validation[column].dropna()) - set(train[column].dropna()))
        for column in ["pickup", "delivery", "equipment"]
    }
    schema_train_only = sorted(set(train.columns) - set(validation.columns))
    schema_validation_only = sorted(set(validation.columns) - set(train.columns))
    quality_checks = pd.DataFrame({
        "train": [
            int(train["load_id"].duplicated().sum()), int(train["date"].isna().sum()),
            int(train["distance"].le(0).sum()), int(train["weight"].lt(0).sum()), int(train[TARGET].le(0).sum()),
        ],
        "validation": [
            int(validation["load_id"].duplicated().sum()), int(validation["date"].isna().sum()),
            int(validation["distance"].le(0).sum()), int(validation["weight"].lt(0).sum()), np.nan,
        ],
    }, index=["duplicate_load_id", "invalid_date_after_parsing", "non_positive_distance", "negative_weight", "non_positive_posted_rate"])
    quality_checks.to_csv(OUTPUT_DIR / "data_quality_checks.csv")
    lines = ["# Phase 2 — Exploratory Data Analysis", "", "All values in this report are calculated from the supplied CSV files."]
    write_section(lines, "Dataset dimensions and dates", "\n".join([
        f"- train-test: {train.shape[0]:,} rows x {train.shape[1]} columns",
        f"- validation: {validation.shape[0]:,} rows x {validation.shape[1]} columns",
        f"- train date range: {train['date'].min().date()} to {train['date'].max().date()} ({train['date'].nunique()} unique dates)",
        f"- validation date range: {validation['date'].min().date()} to {validation['date'].max().date()} ({validation['date'].nunique()} unique dates)",
        "- train rows by month:\n```\n" + format_series(train["date"].dt.to_period("M").value_counts().sort_index()) + "\n```",
        "- validation rows by month:\n```\n" + format_series(validation["date"].dt.to_period("M").value_counts().sort_index()) + "\n```",
    ]))
    write_section(lines, "Data types and missing values", "\n".join([
        "```\nTRAIN DTYPES\n" + format_series(train.dtypes) + "\n```",
        "```\nTRAIN MISSING COUNTS\n" + format_series(train.isna().sum()) + "\n```",
        "```\nVALIDATION MISSING COUNTS\n" + format_series(validation.isna().sum()) + "\n```",
    ]))
    write_section(lines, "Numeric distributions (train)", "```\n" + summary_stats(train, NUMERIC_FEATURES + [TARGET]).to_string() + "\n```")
    write_section(lines, "Categorical distributions", "\n".join([
        "- Train equipment counts:\n```\n" + format_series(train["equipment"].value_counts()) + "\n```",
        "- Validation equipment counts:\n```\n" + format_series(validation["equipment"].value_counts()) + "\n```",
        "- Train top 15 pickup cities:\n```\n" + format_series(train["pickup"].value_counts().head(15)) + "\n```",
        "- Train top 15 delivery cities:\n```\n" + format_series(train["delivery"].value_counts().head(15)) + "\n```",
    ]))
    write_section(lines, "Target and relationships", "\n".join([
        "- Pearson correlation with posted_rate:\n```\n" + format_series(train[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False)) + "\n```",
        "- posted_rate by equipment:\n```\n" + train.groupby("equipment")[TARGET].agg(["count", "mean", "median", "std", "min", "max"]).to_string() + "\n```",
        "- monthly posted_rate:\n```\n" + train.groupby(train["date"].dt.to_period("M"))[TARGET].agg(["count", "mean", "median", "std"]).to_string() + "\n```",
    ]))
    write_section(lines, "Potential outliers (IQR rule; descriptive only)", "```\n" + iqr_outlier_summary(train, NUMERIC_FEATURES + [TARGET]).to_string() + "\n```")
    write_section(lines, "Data-quality checks", "```\n" + quality_checks.to_string() + "\n```")
    write_section(lines, "Train vs validation differences", "\n".join([
        f"- Columns only in train: {schema_train_only}",
        f"- Columns only in validation: {schema_validation_only}",
        f"- Validation-only pickup cities ({len(validation_only['pickup'])}): {validation_only['pickup']}",
        f"- Validation-only delivery cities ({len(validation_only['delivery'])}): {validation_only['delivery']}",
        f"- Validation-only equipment values ({len(validation_only['equipment'])}): {validation_only['equipment']}",
        "- Numeric train/validation comparison (mean and median):\n```\n" + distribution_shift.to_string() + "\n```",
    ]))
    (OUTPUT_DIR / "eda_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created EDA report and tabular outputs in: {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"Created plots in: {PLOTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
