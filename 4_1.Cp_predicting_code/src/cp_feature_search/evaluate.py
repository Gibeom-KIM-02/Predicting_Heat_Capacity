from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def cross_validate_regressor(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    config: Dict[str, Any],
) -> tuple[Dict[str, float], np.ndarray]:
    """
    Return CV metrics and OOF predictions.

    Use GroupKFold if groups have at least 2 unique values.
    Otherwise fall back to normal KFold.
    """
    n_splits = int(config.get("n_splits", 5))
    n_rows = len(X)

    if n_rows < 3:
        raise ValueError(f"Too few rows for CV: n={n_rows}")

    if groups is not None and groups.nunique() >= 2:
        n_groups = groups.nunique()
        cv_n = min(n_splits, n_groups)
        cv_n = max(2, cv_n)
        cv = GroupKFold(n_splits=cv_n)
        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        cv_type = "GroupKFold"
        actual_splits = cv_n
    else:
        cv_n = min(n_splits, n_rows)
        cv_n = max(2, cv_n)
        cv = KFold(n_splits=cv_n, shuffle=True, random_state=int(config.get("random_seed", 42)))
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cv_type = "KFold"
        actual_splits = cv_n

    metrics = metric_row(y, y_pred)
    metrics["cv_type"] = cv_type
    metrics["n_splits"] = int(actual_splits)
    return metrics, y_pred
