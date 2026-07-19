"""Machine learning helpers for stock price prediction and unsupervised analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import DEFAULT_RANDOM_STATE, MODELS_DIR
from .feature_engineering import build_supervised_frame


@dataclass(slots=True)
class RegressionResult:
    model: object
    predictions: np.ndarray
    y_test: np.ndarray
    metrics: dict[str, float]
    feature_columns: list[str]


def _chronological_split(dataframe: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(dataframe) * (1 - test_size))
    train_frame = dataframe.iloc[:split_index].copy()
    test_frame = dataframe.iloc[split_index:].copy()
    return train_frame, test_frame


def _select_feature_columns(dataframe: pd.DataFrame, target_column: str, feature_columns: list[str] | None) -> list[str]:
    if feature_columns:
        return feature_columns
    excluded = {target_column, f"target_{target_column}", "timestamp", "symbol", "day_name", "month_name"}
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded]


def _evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_linear_regression(dataframe: pd.DataFrame, target_column: str = "close", feature_columns: list[str] | None = None, test_size: float = 0.2) -> RegressionResult:
    supervised = build_supervised_frame(dataframe, target_column=target_column)
    features = _select_feature_columns(supervised, target_column=target_column, feature_columns=feature_columns)
    train_frame, test_frame = _chronological_split(supervised, test_size=test_size)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression()),
    ])
    model.fit(train_frame[features], train_frame[target_column])
    predictions = model.predict(test_frame[features])
    metrics = _evaluate_regression(test_frame[target_column], predictions)
    return RegressionResult(model=model, predictions=np.asarray(predictions), y_test=test_frame[target_column].to_numpy(), metrics=metrics, feature_columns=features)


def train_random_forest_regression(dataframe: pd.DataFrame, target_column: str = "close", feature_columns: list[str] | None = None, test_size: float = 0.2, n_estimators: int = 200) -> RegressionResult:
    supervised = build_supervised_frame(dataframe, target_column=target_column)
    features = _select_feature_columns(supervised, target_column=target_column, feature_columns=feature_columns)
    train_frame, test_frame = _chronological_split(supervised, test_size=test_size)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1)
    model.fit(train_frame[features], train_frame[target_column])
    predictions = model.predict(test_frame[features])
    metrics = _evaluate_regression(test_frame[target_column], predictions)
    return RegressionResult(model=model, predictions=np.asarray(predictions), y_test=test_frame[target_column].to_numpy(), metrics=metrics, feature_columns=features)


def train_decision_tree_regression(dataframe: pd.DataFrame, target_column: str = "close", feature_columns: list[str] | None = None, test_size: float = 0.2) -> RegressionResult:
    supervised = build_supervised_frame(dataframe, target_column=target_column)
    features = _select_feature_columns(supervised, target_column=target_column, feature_columns=feature_columns)
    train_frame, test_frame = _chronological_split(supervised, test_size=test_size)

    model = DecisionTreeRegressor(random_state=DEFAULT_RANDOM_STATE)
    model.fit(train_frame[features], train_frame[target_column])
    predictions = model.predict(test_frame[features])
    metrics = _evaluate_regression(test_frame[target_column], predictions)
    return RegressionResult(model=model, predictions=np.asarray(predictions), y_test=test_frame[target_column].to_numpy(), metrics=metrics, feature_columns=features)


def feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    """Return feature importance when the estimator supports it."""

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    importance_frame = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
    return importance_frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path):
    return joblib.load(path)


def detect_outliers_isolation_forest(dataframe: pd.DataFrame, feature_columns: list[str] | None = None, contamination: float = 0.05) -> pd.DataFrame:
    """Label potential anomalies using Isolation Forest."""

    numeric = dataframe.select_dtypes(include=[np.number])
    features = feature_columns or numeric.columns.tolist()
    working = numeric[features].copy().dropna()

    model = IsolationForest(contamination=contamination, random_state=DEFAULT_RANDOM_STATE)
    labels = model.fit_predict(working)
    output = working.copy()
    output["anomaly"] = labels
    return output


def cluster_stocks_kmeans(dataframe: pd.DataFrame, feature_columns: list[str] | None = None, n_clusters: int = 3) -> pd.DataFrame:
    """Cluster stock summaries into similar behavior groups."""

    numeric = dataframe.select_dtypes(include=[np.number])
    features = feature_columns or numeric.columns.tolist()
    working = numeric[features].copy().dropna()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=10)),
    ])
    cluster_labels = pipeline.fit_predict(working)
    output = working.copy()
    output["cluster"] = cluster_labels
    return output


def reduce_dimensions_pca(dataframe: pd.DataFrame, feature_columns: list[str] | None = None, n_components: int = 2) -> pd.DataFrame:
    """Reduce numeric features to two principal components for visualization."""

    numeric = dataframe.select_dtypes(include=[np.number])
    features = feature_columns or numeric.columns.tolist()
    working = numeric[features].copy().dropna()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)),
    ])
    transformed = pipeline.fit_transform(working)
    columns = [f"pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(transformed, columns=columns, index=working.index)
