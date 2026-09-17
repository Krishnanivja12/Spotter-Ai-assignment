from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


IDENTIFIER_COLUMN = "load_id"
TARGET_COLUMN = "posted_rate"
RAW_NUMERIC_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal",
]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
ENGINEERED_NUMERIC_COLUMNS = [
    "weight_was_negative", "distance_log1p", "latitude_difference",
    "longitude_difference", "haversine_distance", "distance_x_market_index",
    "day_of_year_sin", "day_of_year_cos",
]
ENGINEERED_CATEGORICAL_COLUMNS = ["day_of_week"]
MODEL_NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
MODEL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS


class FreightInputAdapter(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FreightInputAdapter":
        frame = X.copy()
        self.pickup_latitude_ = frame.groupby("pickup")["pickup_lat"].median().to_dict()
        self.pickup_longitude_ = frame.groupby("pickup")["pickup_lon"].median().to_dict()
        self.delivery_latitude_ = frame.groupby("delivery")["delivery_lat"].median().to_dict()
        self.delivery_longitude_ = frame.groupby("delivery")["delivery_lon"].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in RAW_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["date"]:
            if column not in frame.columns:
                frame[column] = np.nan
        coordinate_sources = {
            "pickup_lat": ("pickup", self.pickup_latitude_),
            "pickup_lon": ("pickup", self.pickup_longitude_),
            "delivery_lat": ("delivery", self.delivery_latitude_),
            "delivery_lon": ("delivery", self.delivery_longitude_),
        }
        for coordinate, (city_column, lookup) in coordinate_sources.items():
            frame[coordinate] = frame[coordinate].fillna(frame[city_column].map(lookup))
        return frame


class PositiveRidgeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, alpha: float = 10.0, prediction_floor: float = 0.01) -> None:
        self.alpha = alpha
        self.prediction_floor = prediction_floor

    def fit(self, X: object, y: pd.Series) -> "PositiveRidgeRegressor":
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, y)
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.maximum(self.model_.predict(X), self.prediction_floor)


class FreightDataCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FreightDataCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        required = set(RAW_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["date"])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        frame["weight_was_negative"] = frame["weight"].lt(0).astype("int8")
        frame.loc[frame["weight"].lt(0), "weight"] = np.nan
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame


class FreightFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create compact, deterministic features that are known at prediction time."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FreightFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame = frame.drop(columns=[IDENTIFIER_COLUMN, TARGET_COLUMN], errors="ignore")
        date = pd.to_datetime(frame["date"], errors="coerce")
        lat_1 = np.radians(frame["pickup_lat"])
        lon_1 = np.radians(frame["pickup_lon"])
        lat_2 = np.radians(frame["delivery_lat"])
        lon_2 = np.radians(frame["delivery_lon"])
        haversine_term = np.sin((lat_2 - lat_1) / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin((lon_2 - lon_1) / 2) ** 2

        frame["distance_log1p"] = np.log1p(frame["distance"])
        frame["latitude_difference"] = frame["delivery_lat"] - frame["pickup_lat"]
        frame["longitude_difference"] = frame["delivery_lon"] - frame["pickup_lon"]
        frame["haversine_distance"] = 3958.7613 * 2 * np.arcsin(np.sqrt(haversine_term))
        frame["distance_x_market_index"] = frame["distance"] * frame["market_index"]
        frame["day_of_week"] = date.dt.day_name()
        angle = 2 * np.pi * date.dt.dayofyear / 365.25
        frame["day_of_year_sin"] = np.sin(angle)
        frame["day_of_year_cos"] = np.cos(angle)
        return frame.drop(columns=["date"])


def build_model_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    scale_numeric: bool = False,
    dense_output: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        sparse_threshold=0 if dense_output else 0.3,
    )


def build_feature_pipeline() -> Pipeline:
    """Return the one reusable cleaning, engineering, and encoding pipeline."""
    return Pipeline(
        steps=[
            ("adapt_inputs", FreightInputAdapter()),
            ("clean", FreightDataCleaner()),
            ("features", FreightFeatureEngineer()),
            ("preprocess", build_model_preprocessor(MODEL_NUMERIC_COLUMNS, MODEL_CATEGORICAL_COLUMNS)),
        ]
    )


def build_final_ridge_pipeline(alpha: float = 10.0, prediction_floor: float = 0.01) -> Pipeline:
    return Pipeline(
        steps=[
            ("adapt_inputs", FreightInputAdapter()),
            ("clean", FreightDataCleaner()),
            ("features", FreightFeatureEngineer()),
            (
                "preprocess",
                build_model_preprocessor(
                    MODEL_NUMERIC_COLUMNS,
                    MODEL_CATEGORICAL_COLUMNS,
                    scale_numeric=True,
                    dense_output=True,
                ),
            ),
            ("model", PositiveRidgeRegressor(alpha=alpha, prediction_floor=prediction_floor)),
        ]
    )
