from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_regressor(model_name: str, config: Dict[str, Any]) -> Pipeline:
    """
    Build regression model for feature search.
    """
    random_seed = int(config.get("random_seed", 42))

    if model_name == "linear":
        model = LinearRegression()
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]

    elif model_name == "ridge":
        model = Ridge(alpha=1.0, random_state=random_seed)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]

    elif model_name == "elasticnet":
        model = ElasticNet(
            alpha=0.01,
            l1_ratio=0.3,
            max_iter=100000,
            tol=1e-3,
            random_state=random_seed,
            selection="random",
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]

    elif model_name == "svr":
        model = SVR(C=10.0, epsilon=0.1, kernel="rbf")
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]

    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=2,
            random_state=random_seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]

    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    return Pipeline(steps)
