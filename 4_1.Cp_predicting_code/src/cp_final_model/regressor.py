from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, train_test_split

from .data_prep import make_training_dataset
from .models import build_cp_regressor


# =============================================================================
# Basic metrics
# =============================================================================

def _rmse(y_true, y_pred) -> float:
    """Return root-mean-square error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metric_dict(y_true, y_pred, prefix: str) -> Dict[str, Any]:
    """Return MAE / RMSE / R2 metrics with a prefix."""
    return {
        f"{prefix}_n": int(len(y_true)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": _rmse(y_true, y_pred),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


# =============================================================================
# CV and split helpers
# =============================================================================

def _make_cv(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Build a CV splitter.

    Preferred:
        GroupKFold by canonical_smiles to avoid putting the same compound
        in both train and validation folds.

    Fallback:
        KFold if group information is not usable.
    """
    n_splits = int(config.get("n_splits", 5))
    groups = df["canonical_smiles"].astype(str)

    if groups.nunique() >= 2:
        cv_n = min(n_splits, groups.nunique())
        cv_n = max(2, cv_n)
        return GroupKFold(n_splits=cv_n), groups, "GroupKFold", cv_n

    cv_n = min(n_splits, len(df))
    cv_n = max(2, cv_n)

    return (
        KFold(
            n_splits=cv_n,
            shuffle=True,
            random_state=int(config.get("random_seed", 42)),
        ),
        None,
        "KFold",
        cv_n,
    )


def _make_group_level_table(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Build one row per canonical_smiles for group-level splitting.

    Each compound group receives one representative Cp value.
    This representative value is used only for stratifying Cp distribution.
    """
    group_df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(Cp_group=("Cp_J_molK", "mean"))
    )

    n_bins = max(2, min(int(n_bins), len(group_df)))

    group_df["Cp_bin"] = pd.qcut(
        group_df["Cp_group"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    group_df["Cp_bin"] = group_df["Cp_bin"].fillna(0).astype(int)
    return group_df


def _safe_group_table_for_stratify(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Build a group-level table whose Cp_bin is safe for stratified splitting.

    train_test_split(stratify=...) fails if any bin has only one group.
    This function reduces the number of bins until every bin has at least two groups.
    """
    n_bins = int(n_bins)

    while True:
        group_df = _make_group_level_table(df, n_bins=n_bins)
        counts = group_df["Cp_bin"].value_counts()

        if group_df["Cp_bin"].nunique() <= 1:
            group_df["Cp_bin"] = 0
            return group_df

        if counts.min() >= 2:
            return group_df

        n_bins -= 1
        if n_bins < 2:
            group_df["Cp_bin"] = 0
            return group_df


def _make_train_val_test_split(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split into train / validation / test by canonical_smiles groups.

    This prevents compound leakage across splits and approximately preserves
    the Cp distribution using quantile bins at the compound-group level.
    """
    split_cfg = config.get("split", {})

    train_size = float(split_cfg.get("train_size", 0.70))
    val_size = float(split_cfg.get("val_size", 0.15))
    test_size = float(split_cfg.get("test_size", 0.15))
    n_bins = int(split_cfg.get("n_bins", 10))
    random_seed = int(split_cfg.get("random_seed", config.get("random_seed", 42)))

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size must be 1.0, but got {total}"
        )

    group_df = _safe_group_table_for_stratify(df, n_bins=n_bins)

    stratify_main = (
        group_df["Cp_bin"] if group_df["Cp_bin"].nunique() > 1 else None
    )

    # First split: train vs temp(val+test)
    train_groups, temp_groups = train_test_split(
        group_df["canonical_smiles"],
        train_size=train_size,
        random_state=random_seed,
        stratify=stratify_main,
    )

    temp_df = group_df[group_df["canonical_smiles"].isin(set(temp_groups))].copy()

    # Second split: validation vs test within temp.
    # val fraction inside temp = val_size / (val_size + test_size)
    val_fraction_in_temp = val_size / (val_size + test_size)

    temp_stratify = (
        temp_df["Cp_bin"] if temp_df["Cp_bin"].nunique() > 1 and temp_df["Cp_bin"].value_counts().min() >= 2 else None
    )

    val_groups, test_groups = train_test_split(
        temp_df["canonical_smiles"],
        train_size=val_fraction_in_temp,
        random_state=random_seed,
        stratify=temp_stratify,
    )

    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    train_df = df[df["canonical_smiles"].isin(train_set)].copy()
    val_df = df[df["canonical_smiles"].isin(val_set)].copy()
    test_df = df[df["canonical_smiles"].isin(test_set)].copy()

    split_info = {
        "split_train_rows": int(len(train_df)),
        "split_val_rows": int(len(val_df)),
        "split_test_rows": int(len(test_df)),
        "split_train_unique_compounds": int(train_df["canonical_smiles"].nunique()),
        "split_val_unique_compounds": int(val_df["canonical_smiles"].nunique()),
        "split_test_unique_compounds": int(test_df["canonical_smiles"].nunique()),
        "split_train_size_requested": train_size,
        "split_val_size_requested": val_size,
        "split_test_size_requested": test_size,
    }

    return train_df, val_df, test_df, split_info


# =============================================================================
# Sample weighting
# =============================================================================

def _make_cp_bin_weights(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """
    Build inverse-frequency sample weights based on Cp quantile bins.

    Dense Cp regions receive lower weights.
    Sparse Cp regions receive higher weights.
    """
    n_bins = max(2, min(int(n_bins), len(y)))

    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    bins = pd.Series(bins).fillna(0).astype(int)

    counts = bins.value_counts().to_dict()
    weights = bins.map(lambda b: 1.0 / counts[int(b)]).astype(float).values

    weights = weights / np.mean(weights)
    return weights


def _sample_weight_enabled(config: Dict[str, Any]) -> bool:
    """Return whether sample weighting is enabled."""
    return bool(config.get("sample_weighting", {}).get("enabled", False))


def _get_sample_weight(y: pd.Series, config: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Return sample weights if enabled.

    Supported method:
        cp_bin_inverse_frequency
    """
    cfg = config.get("sample_weighting", {})
    if not bool(cfg.get("enabled", False)):
        return None

    method = cfg.get("method", "cp_bin_inverse_frequency")
    if method != "cp_bin_inverse_frequency":
        raise ValueError(f"Unsupported sample_weighting method: {method}")

    n_bins = int(cfg.get("n_bins", 10))
    return _make_cp_bin_weights(y, n_bins=n_bins)


def _fit_model(model, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]):
    """
    Fit a sklearn Pipeline, optionally with sample weights.

    The final estimator step is named 'model' in build_cp_regressor().
    Therefore sample_weight must be passed as model__sample_weight.
    """
    sample_weight = _get_sample_weight(y, config)

    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, model__sample_weight=sample_weight)

    return model


# =============================================================================
# Plotting and error analysis
# =============================================================================

def _save_parity_plot(y_true, y_pred, outpath: Path, title: str) -> None:
    """Save a parity plot."""
    outpath.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    mae = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan

    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("True Cp [J/mol*K]")
    plt.ylabel("Predicted Cp [J/mol*K]")
    plt.title(title)

    metric_text = (
        f"MAE  = {mae:.3f} J/mol*K\n"
        f"RMSE = {rmse:.3f} J/mol*K\n"
        f"R²   = {r2:.4f}\n"
        f"N    = {len(y_true)}"
    )

    plt.text(
        0.05,
        0.95,
        metric_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def _save_error_by_cp_bin(
    pred_df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    n_bins: int = 10,
) -> None:
    """
    Save error statistics by Cp quantile bin.

    Required columns:
        Cp_J_molK
        Cp_pred
        abs_error
    """
    out = pred_df.copy()

    required = {"Cp_J_molK", "Cp_pred", "abs_error"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns for Cp-bin error analysis: {missing}")

    if len(out) < 2:
        return

    n_bins = max(2, min(int(n_bins), len(out)))

    out["Cp_bin"] = pd.qcut(
        out["Cp_J_molK"],
        q=n_bins,
        duplicates="drop",
    )

    rows = []
    for bin_label, sub in out.groupby("Cp_bin", observed=False):
        y_true = sub["Cp_J_molK"].to_numpy(dtype=float)
        y_pred = sub["Cp_pred"].to_numpy(dtype=float)

        rows.append(
            {
                "Cp_bin": str(bin_label),
                "n": int(len(sub)),
                "Cp_min": float(sub["Cp_J_molK"].min()),
                "Cp_max": float(sub["Cp_J_molK"].max()),
                "Cp_mean": float(sub["Cp_J_molK"].mean()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": _rmse(y_true, y_pred),
                "mean_abs_error": float(sub["abs_error"].mean()),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / filename, index=False)


def _save_feature_importance_plot(
    model,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Save RandomForest/GradientBoosting feature importance table and plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        print("[WARN] Final estimator does not provide feature_importances_. Skipping.")
        return

    fi = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    fi.to_csv(output_dir / "feature_importance_all.csv", index=False)
    fi.head(top_n).to_csv(output_dir / f"feature_importance_top{top_n}.csv", index=False)

    plot_df = fi.head(top_n).iloc[::-1].copy()

    plt.figure(figsize=(7.0, max(4.5, 0.32 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_top{top_n}.png", dpi=300)
    plt.close()


def _save_training_log_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save RandomForest performance as n_estimators increases.

    This is not an epoch curve, but it is the closest meaningful training-size
    curve for RandomForest.
    """
    log_cfg = config.get("training_log", {})
    log_dir = Path(log_cfg.get("output_dir", output_dir / "training_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    grid = log_cfg.get("rf_n_estimators_grid", [50, 100, 200, 400, 800])

    X_train = train_df[feature_names]
    y_train = train_df["Cp_J_molK"].astype(float)

    X_val = val_df[feature_names]
    y_val = val_df["Cp_J_molK"].astype(float)

    X_test = test_df[feature_names]
    y_test = test_df["Cp_J_molK"].astype(float)

    rows = []

    base_cfg = dict(config)
    base_reg_cfg = dict(config.get("regressor", {}))

    for n_estimators in grid:
        cfg = dict(base_cfg)
        reg_cfg = dict(base_reg_cfg)
        reg_cfg["model_type"] = "random_forest"
        reg_cfg["n_estimators"] = int(n_estimators)
        cfg["regressor"] = reg_cfg

        model = build_cp_regressor(cfg)
        model = _fit_model(model, X_train, y_train, cfg)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        row = {
            "model_type": "random_forest",
            "n_estimators": int(n_estimators),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "train_rmse": _rmse(y_train, y_train_pred),
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "validation_mae": float(mean_absolute_error(y_val, y_val_pred)),
            "validation_rmse": _rmse(y_val, y_val_pred),
            "validation_r2": float(r2_score(y_val, y_val_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
            "test_rmse": _rmse(y_test, y_test_pred),
            "test_r2": float(r2_score(y_test, y_test_pred)),
        }
        rows.append(row)

    log_df = pd.DataFrame(rows)
    log_df.to_csv(log_dir / "random_forest_n_estimators_curve.csv", index=False)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(log_df["n_estimators"], log_df["train_mae"], marker="o", label="train")
    plt.plot(log_df["n_estimators"], log_df["validation_mae"], marker="o", label="validation")
    plt.plot(log_df["n_estimators"], log_df["test_mae"], marker="o", label="test")
    plt.xlabel("n_estimators")
    plt.ylabel("MAE [J/mol*K]")
    plt.title("RandomForest n_estimators Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "random_forest_n_estimators_curve.png", dpi=300)
    plt.close()


def _save_training_log_gradient_boosting(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save GradientBoosting staged MAE curve.

    For GradientBoostingRegressor, each staged_predict step corresponds to one
    additional boosting estimator. This is the closest sklearn equivalent of an
    epoch-wise learning curve.
    """
    log_cfg = config.get("training_log", {})
    log_dir = Path(log_cfg.get("output_dir", output_dir / "training_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_names]
    y_train = train_df["Cp_J_molK"].astype(float)

    X_val = val_df[feature_names]
    y_val = val_df["Cp_J_molK"].astype(float)

    X_test = test_df[feature_names]
    y_test = test_df["Cp_J_molK"].astype(float)

    cfg = dict(config)
    reg_cfg = dict(config.get("regressor", {}))
    reg_cfg["model_type"] = "gradient_boosting"
    cfg["regressor"] = reg_cfg

    model = build_cp_regressor(cfg)
    model = _fit_model(model, X_train, y_train, cfg)

    gb = model.named_steps["model"]

    if not hasattr(gb, "staged_predict"):
        return

    rows = []

    for stage_idx, (pred_train, pred_val, pred_test) in enumerate(
        zip(
            gb.staged_predict(model.named_steps["imputer"].transform(X_train)),
            gb.staged_predict(model.named_steps["imputer"].transform(X_val)),
            gb.staged_predict(model.named_steps["imputer"].transform(X_test)),
        ),
        start=1,
    ):
        rows.append(
            {
                "model_type": "gradient_boosting",
                "stage": int(stage_idx),
                "learning_rate": float(reg_cfg.get("learning_rate", 0.03)),
                "train_mae": float(mean_absolute_error(y_train, pred_train)),
                "validation_mae": float(mean_absolute_error(y_val, pred_val)),
                "test_mae": float(mean_absolute_error(y_test, pred_test)),
            }
        )

    log_df = pd.DataFrame(rows)
    log_df.to_csv(log_dir / "gradient_boosting_staged_mae.csv", index=False)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(log_df["stage"], log_df["train_mae"], label="train")
    plt.plot(log_df["stage"], log_df["validation_mae"], label="validation")
    plt.plot(log_df["stage"], log_df["test_mae"], label="test")
    plt.xlabel("Boosting stage")
    plt.ylabel("MAE [J/mol*K]")
    plt.title("GradientBoosting Staged MAE Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "gradient_boosting_staged_mae.png", dpi=300)
    plt.close()


def _safe_meta_columns(df: pd.DataFrame) -> list[str]:
    """Return metadata columns that exist in df."""
    candidate_cols = [
        "compound_name",
        "canonical_smiles",
        "smiles_raw",
        "Cp_J_molK",
        "source",
        "source_sheet",
    ]
    return [col for col in candidate_cols if col in df.columns]


def _make_prediction_table(
    df: pd.DataFrame,
    y_pred,
    split_name: str,
) -> pd.DataFrame:
    """Build a standardized prediction table."""
    out = df[_safe_meta_columns(df)].copy()
    out["split"] = split_name
    out["Cp_pred"] = np.asarray(y_pred, dtype=float)
    out["abs_error"] = np.abs(out["Cp_J_molK"] - out["Cp_pred"])
    return out


def _save_feature_importance_plot(
    model,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Save RandomForest/GradientBoosting feature importance table and plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        print("[WARN] Final estimator does not provide feature_importances_. Skipping.")
        return

    fi = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    fi.to_csv(output_dir / "feature_importance_all.csv", index=False)
    fi.head(top_n).to_csv(output_dir / f"feature_importance_top{top_n}.csv", index=False)

    plot_df = fi.head(top_n).iloc[::-1].copy()

    plt.figure(figsize=(7.0, max(4.5, 0.32 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_top{top_n}.png", dpi=300)
    plt.close()


def _save_shap_analysis(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    max_samples: int = 500,
    random_seed: int = 42,
) -> None:
    """
    Save SHAP analysis for the final tree-based model.

    This function is optional. If shap is not installed, it prints a warning
    and does not stop the main training pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except Exception as exc:
        print(f"[WARN] shap is not available. Skipping SHAP analysis. Reason: {exc}")
        return

    if len(X) == 0:
        print("[WARN] Empty X passed to SHAP analysis. Skipping.")
        return

    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=random_seed).copy()
    else:
        X_sample = X.copy()

    # The saved model is a Pipeline: imputer + model.
    imputer = model.named_steps.get("imputer")
    estimator = model.named_steps["model"]

    if imputer is not None:
        X_imp = imputer.transform(X_sample)
    else:
        X_imp = X_sample.values

    X_imp_df = pd.DataFrame(X_imp, columns=feature_names, index=X_sample.index)

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_imp_df)
    except Exception as exc:
        print(f"[WARN] SHAP TreeExplainer failed. Skipping SHAP analysis. Reason: {exc}")
        return

    # Save mean absolute SHAP importance.
    shap_arr = np.asarray(shap_values)

    # Regression models usually return shape = (n_samples, n_features).
    # Some SHAP versions may return a list; handle basic case.
    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[0])

    mean_abs = np.abs(shap_arr).mean(axis=0)

    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(output_dir / "shap_mean_abs_importance.csv", index=False)

    shap_values_df = pd.DataFrame(
        shap_arr,
        columns=[f"SHAP_{name}" for name in feature_names],
        index=X_sample.index,
    )
    shap_values_df.to_csv(output_dir / "shap_values_sample.csv", index=True)

    # SHAP bar plot
    try:
        plt.figure()
        shap.summary_plot(
            shap_arr,
            X_imp_df,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[WARN] Failed to save SHAP bar plot. Reason: {exc}")

    # SHAP beeswarm plot
    try:
        plt.figure()
        shap.summary_plot(
            shap_arr,
            X_imp_df,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[WARN] Failed to save SHAP beeswarm plot. Reason: {exc}")


# =============================================================================
# Main training entry point
# =============================================================================

def train_final_regressor(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train one compliant Cp regression model.

    No phase labels are used.

    Input features:
        1. SMILES-derived descriptors
        2. optional <=2 Excel-provided physical features

    Evaluation:
        1. GroupKFold OOF CV
        2. Optional train / validation / test split

    Final saved model:
        Controlled by split.refit_final_on
        - full: train final deployable model on all cleaned data
        - train_val: train final deployable model on train + validation only
    """
    output_dir = Path(config["output_dir"])
    model_dir = Path(config["model_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df, feature_names = make_training_dataset(config)

    if len(df) < 5:
        raise ValueError(f"Too few valid rows for training: n={len(df)}")

    X = df[feature_names]
    y = df["Cp_J_molK"].astype(float)

    # -------------------------------------------------------------------------
    # OOF CV prediction
    # -------------------------------------------------------------------------
    cv_model = build_cp_regressor(config)
    cv, groups, cv_type, cv_n = _make_cv(df, config)

    if groups is not None:
        y_oof = cross_val_predict(cv_model, X, y, cv=cv, groups=groups)
    else:
        y_oof = cross_val_predict(cv_model, X, y, cv=cv)

    mae = float(mean_absolute_error(y, y_oof))
    rmse = _rmse(y, y_oof)
    r2 = float(r2_score(y, y_oof)) if len(df) >= 2 else np.nan

    oof_df = _make_prediction_table(df, y_oof, split_name="oof_cv")
    oof_df = oof_df.rename(columns={"Cp_pred": "Cp_pred_oof"})
    oof_df["Cp_pred"] = oof_df["Cp_pred_oof"]
    oof_df.to_csv(output_dir / "final_regressor_oof_predictions.csv", index=False)

    _save_parity_plot(
        y,
        y_oof,
        output_dir / "parity_oof_cv.png",
        "OOF Cross-Validated Cp Prediction",
    )

    # Avoid duplicate Cp_pred columns: use a clean table with only Cp_pred.
    oof_bin_df = _make_prediction_table(df, y_oof, split_name="oof_cv")
    _save_error_by_cp_bin(
        oof_bin_df,
        output_dir,
        filename="oof_error_by_cp_bin.csv",
        n_bins=int(config.get("error_analysis", {}).get("n_bins", 10)),
    )


    # -------------------------------------------------------------------------
    # Optional train / validation / test split
    # -------------------------------------------------------------------------
    split_summary: Dict[str, Any] = {}
    refit_final_on = config.get("split", {}).get("refit_final_on", "full")

    final_fit_df = df.copy()

    split_enabled = bool(config.get("split", {}).get("enabled", False))
    training_log_enabled = bool(config.get("training_log", {}).get("enabled", False))

    if training_log_enabled and not split_enabled:
        print("[WARN] training_log.enabled=true but split.enabled=false. Skipping training logs.")

    if split_enabled:
        train_df, val_df, test_df, split_info = _make_train_val_test_split(df, config)

        X_train = train_df[feature_names]
        y_train = train_df["Cp_J_molK"].astype(float)

        X_val = val_df[feature_names]
        y_val = val_df["Cp_J_molK"].astype(float)

        X_test = test_df[feature_names]
        y_test = test_df["Cp_J_molK"].astype(float)

        split_model = build_cp_regressor(config)
        split_model = _fit_model(split_model, X_train, y_train, config)

        y_train_pred = split_model.predict(X_train)
        y_val_pred = split_model.predict(X_val)
        y_test_pred = split_model.predict(X_test)

        train_pred_df = _make_prediction_table(train_df, y_train_pred, "train")
        val_pred_df = _make_prediction_table(val_df, y_val_pred, "validation")
        test_pred_df = _make_prediction_table(test_df, y_test_pred, "test")

        split_pred_df = pd.concat(
            [train_pred_df, val_pred_df, test_pred_df],
            ignore_index=True,
        )
        split_pred_df.to_csv(output_dir / "train_validation_test_predictions.csv", index=False)

        _save_parity_plot(
            y_train,
            y_train_pred,
            output_dir / "parity_train_set.png",
            "Train Set Cp Prediction",
        )
        _save_parity_plot(
            y_val,
            y_val_pred,
            output_dir / "parity_validation_set.png",
            "Validation Set Cp Prediction",
        )
        _save_parity_plot(
            y_test,
            y_test_pred,
            output_dir / "parity_test_set.png",
            "Test Set Cp Prediction",
        )

        n_error_bins = int(config.get("error_analysis", {}).get("n_bins", 10))
        _save_error_by_cp_bin(
            train_pred_df,
            output_dir,
            filename="train_error_by_cp_bin.csv",
            n_bins=n_error_bins,
        )
        _save_error_by_cp_bin(
            val_pred_df,
            output_dir,
            filename="validation_error_by_cp_bin.csv",
            n_bins=n_error_bins,
        )
        _save_error_by_cp_bin(
            test_pred_df,
            output_dir,
            filename="test_error_by_cp_bin.csv",
            n_bins=n_error_bins,
        )

        # -------------------------------------------------------------
        # Training log curves
        # This must be inside split_enabled because it needs train/val/test.
        # -------------------------------------------------------------
        if training_log_enabled:
            model_type = config.get("regressor", {}).get("model_type", "random_forest")

            if model_type == "random_forest":
                _save_training_log_random_forest(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    feature_names=feature_names,
                    config=config,
                    output_dir=output_dir,
                )

            elif model_type == "gradient_boosting":
                _save_training_log_gradient_boosting(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    feature_names=feature_names,
                    config=config,
                    output_dir=output_dir,
                )

        split_summary.update(split_info)
        split_summary.update(_metric_dict(y_train, y_train_pred, "train"))
        split_summary.update(_metric_dict(y_val, y_val_pred, "validation"))
        split_summary.update(_metric_dict(y_test, y_test_pred, "test"))

        if refit_final_on == "full":
            final_fit_df = df.copy()
        elif refit_final_on == "train_val":
            final_fit_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            raise ValueError(
                f"Unsupported split.refit_final_on={refit_final_on}. "
                "Use 'full' or 'train_val'."
            )

    # -------------------------------------------------------------------------
    # Final deployable model fit
    # -------------------------------------------------------------------------
    X_final = final_fit_df[feature_names]
    y_final = final_fit_df["Cp_J_molK"].astype(float)

    final_model = build_cp_regressor(config)
    final_model = _fit_model(final_model, X_final, y_final, config)

    joblib.dump(final_model, model_dir / "cp_regressor.joblib")

    schema = {
        "model_file": "cp_regressor.joblib",
        "feature_set": config["feature_set"],
        "feature_names": feature_names,
        "extra_feature_columns": config.get("extra_feature_columns", []) or [],
        "target": "Cp_J_molK",
        "unit": "J/mol*K",
        "cv_type": cv_type,
        "n_splits": int(cv_n),
        "sample_weighting": config.get("sample_weighting", {"enabled": False}),
        "split": config.get("split", {"enabled": False}),
        "refit_final_on": refit_final_on,
        "n_final_fit_rows": int(len(final_fit_df)),
        "n_final_fit_unique_compounds": int(final_fit_df["canonical_smiles"].nunique()),
    }

    with open(model_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    summary = {
        "n_rows": int(len(df)),
        "n_unique_compounds": int(df["canonical_smiles"].nunique()),
        "n_features": int(len(feature_names)),
        "cv_type": cv_type,
        "n_splits": int(cv_n),
        "mae_oof": mae,
        "rmse_oof": rmse,
        "r2_oof": r2,
        "sample_weighting_enabled": _sample_weight_enabled(config),
        **split_summary,
        "refit_final_on": refit_final_on,
        "n_final_fit_rows": int(len(final_fit_df)),
        "n_final_fit_unique_compounds": int(final_fit_df["canonical_smiles"].nunique()),
        "model_path": str(model_dir / "cp_regressor.joblib"),
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "final_regressor_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Interpretability: feature importance and optional SHAP
    # -------------------------------------------------------------------------
    interpret_cfg = config.get("interpretability", {})
    interpret_enabled = bool(interpret_cfg.get("enabled", False))

    # Keep the old simple CSV for compatibility.
    estimator = final_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        fi.to_csv(output_dir / "final_regressor_feature_importance.csv", index=False)

    if interpret_enabled:
        interpret_dir = Path(
            interpret_cfg.get(
                "output_dir",
                output_dir / "interpretability",
            )
        )
        interpret_dir.mkdir(parents=True, exist_ok=True)

        if bool(interpret_cfg.get("save_feature_importance_plot", True)):
            _save_feature_importance_plot(
                model=final_model,
                feature_names=feature_names,
                output_dir=interpret_dir,
                top_n=int(interpret_cfg.get("top_n", 20)),
            )

        if bool(interpret_cfg.get("save_shap", True)):
            _save_shap_analysis(
                model=final_model,
                X=X_final,
                feature_names=feature_names,
                output_dir=interpret_dir,
                max_samples=int(interpret_cfg.get("max_samples", 500)),
                random_seed=int(interpret_cfg.get("random_seed", config.get("random_seed", 42))),
            )

    return summary