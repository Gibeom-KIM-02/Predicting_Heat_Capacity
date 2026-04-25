from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_phase_classifier(config: Dict[str, Any]) -> Pipeline:
    """Build the phase routing classifier."""
    cfg = config.get("phase_classifier", {})
    random_seed = int(config.get("random_seed", 42))

    model_type = cfg.get("model_type", "random_forest")
    if model_type != "random_forest":
        raise ValueError(f"Unsupported phase classifier model_type={model_type}")

    model = RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 500)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _none_if_null(value):
    """Convert YAML null / 'none' / 'null' to Python None."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return value


def build_cp_regressor(config: Dict[str, Any]) -> Pipeline:
    """Build the final Cp regressor."""
    cfg = config.get("regressor", {})
    random_seed = int(config.get("random_seed", 42))

    model_type = cfg.get("model_type", "random_forest")

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(cfg.get("n_estimators", 800)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
            min_samples_split=int(cfg.get("min_samples_split", 2)),
            max_depth=_none_if_null(cfg.get("max_depth", None)),
            max_features=cfg.get("max_features", "sqrt"),
            bootstrap=bool(cfg.get("bootstrap", True)),
            random_state=random_seed,
            n_jobs=-1,
        )

    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(cfg.get("n_estimators", 500)),
            learning_rate=float(cfg.get("learning_rate", 0.03)),
            max_depth=int(cfg.get("max_depth", 3)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            min_samples_split=int(cfg.get("min_samples_split", 2)),
            subsample=float(cfg.get("subsample", 1.0)),
            random_state=random_seed,
        )

    else:
        raise ValueError(f"Unsupported regressor model_type={model_type}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

