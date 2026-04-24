#!/usr/bin/env python3
"""
Feature-pair selector for Cp prediction with:
- regression screening
- information gain diagnostics
- correlation / redundancy diagnostics
- holdout evaluation for the final selected pair
- SHAP interpretation
- publication-style visualizations

This script is intentionally focused on selecting the best two physical
features under the assignment constraint, while still keeping several
useful outputs from the original modeling script:
    - best_model_holdout_parity.png
    - best_model_residual_plot.png
    - best_model_shap_bar.png
    - best_model_shap_beeswarm.png
    - best_model__dep__<feature>.png

Typical usage
-------------
python 01_feature_pair_selector_with_holdout.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


DEFAULT_CANDIDATE_FEATURES = [
    "molecular_weight",
    "rotatable_bonds",
    "H_bond_donors",
    "H_bond_acceptors",
    "TPSA",
    "logP",
    "melting_point",
    "boiling_point",
]

DEFAULT_MODELS = ["linear", "ridge", "elasticnet", "svr", "rf", "gbr"]


@dataclass
class FeatureSelectorConfig:
    config_file: Optional[str] = None

    # I/O
    data_file: str = "mid_project_verified.xlsx"
    sheet_name: Optional[str] = None
    output_dir: str = "FEATURE_SELECTION_RESULTS"

    # columns
    smiles_col: str = "smiles"
    name_col: str = "name"
    target_col: str = "heat_capacity"
    group_col: str = "group"

    # input cleanup
    skip_unit_row: bool = True
    auto_drop_unit_row: bool = True

    # feature candidates
    candidate_features: Sequence[str] = tuple(DEFAULT_CANDIDATE_FEATURES)
    fixed_pairs: Optional[Sequence[str]] = None

    # filtering
    min_non_null_ratio: float = 0.7
    min_pair_rows: int = 30
    low_variance_threshold: float = 0.0
    iqr_outlier_threshold: float = 0.0
    redundancy_corr_threshold: float = 0.85

    # information gain
    use_information_gain: bool = True
    target_bins: int = 3
    feature_bins: int = 4

    # modeling
    models: Sequence[str] = tuple(DEFAULT_MODELS)
    cv_splits: int = 5
    cv_repeats: int = 10
    test_size: float = 0.2
    seed: int = 2357
    top_k_pairs: int = 10
    model_params: Optional[Dict[str, Dict[str, Any]]] = None

    # plots
    save_plots: bool = True
    top_n_plot: int = 15
    save_distribution_plots: bool = True
    save_violin_plot: bool = True
    save_correlation_heatmap: bool = True
    save_ig_plots: bool = True
    save_holdout_plots: bool = True

    # shap / explanation
    no_shap: bool = False
    save_shap_beeswarm: bool = True
    save_shap_dependence: bool = True


@dataclass
class HoldoutResult:
    model_name: str
    feature_pair: Tuple[str, str]
    target_col: str
    mae_train: float
    mae_test: float
    rmse_test: float
    r2_test: float
    fitted_pipeline: Pipeline
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class CpFeaturePairSelector:
    def __init__(self, config: FeatureSelectorConfig):
        self.cfg = config
        self.data_path = Path(config.data_file).resolve()
        self.out_dir = Path(config.output_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

        self.retained_features: List[str] = []
        self.feature_pairs: List[Tuple[str, str]] = []

        self.pipelines: Dict[str, Pipeline] = self._build_model_pipelines()

        self.feature_quality_df: Optional[pd.DataFrame] = None
        self.single_feature_ranking_df: Optional[pd.DataFrame] = None
        self.pair_ranking_df: Optional[pd.DataFrame] = None
        self.ig_df: Optional[pd.DataFrame] = None

        self._set_seed()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _set_seed(self) -> None:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def _build_model_pipelines(self) -> Dict[str, Pipeline]:
        scaled = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        unscaled = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        model_params = self.cfg.model_params or {}

        ridge_params: Dict[str, Any] = {
            "alpha": 1.0,
            "random_state": self.cfg.seed,
        }
        ridge_params.update(model_params.get("ridge", {}))

        elasticnet_params: Dict[str, Any] = {
            "alpha": 0.01,
            "l1_ratio": 0.3,
            "random_state": self.cfg.seed,
            "max_iter": 20000,
        }
        elasticnet_params.update(model_params.get("elasticnet", {}))

        svr_params: Dict[str, Any] = {
            "kernel": "rbf",
            "C": 10.0,
            "epsilon": 0.1,
            "gamma": "scale",
        }
        svr_params.update(model_params.get("svr", {}))

        rf_params: Dict[str, Any] = {
            "n_estimators": 500,
            "random_state": self.cfg.seed,
            "n_jobs": -1,
            "min_samples_leaf": 2,
        }
        rf_params.update(model_params.get("rf", {}))

        gbr_params: Dict[str, Any] = {
            "random_state": self.cfg.seed,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 2,
            "subsample": 0.9,
        }
        gbr_params.update(model_params.get("gbr", {}))

        return {
            "linear": Pipeline([("prep", scaled), ("model", LinearRegression())]),
            "ridge": Pipeline([("prep", scaled), ("model", Ridge(**ridge_params))]),
            "elasticnet": Pipeline([("prep", scaled), ("model", ElasticNet(**elasticnet_params))]),
            "svr": Pipeline([("prep", scaled), ("model", SVR(**svr_params))]),
            "rf": Pipeline([("prep", unscaled), ("model", RandomForestRegressor(**rf_params))]),
            "gbr": Pipeline([("prep", unscaled), ("model", GradientBoostingRegressor(**gbr_params))]),
        }

    # ------------------------------------------------------------------
    # data loading / cleaning
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        if self.cfg.sheet_name is None:
            df = pd.read_excel(self.data_path)
        else:
            df = pd.read_excel(self.data_path, sheet_name=self.cfg.sheet_name)

        if self.cfg.skip_unit_row:
            df = self._drop_unit_row(df)

        self.raw_df = df.reset_index(drop=True)

    def _drop_unit_row(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if not self.cfg.auto_drop_unit_row:
            return df.iloc[1:].reset_index(drop=True)

        first = df.iloc[0].astype(str).str.strip()

        unit_like_count = 0
        for value in first.values:
            if value in {"-", "—", "g/mol", "J/mol·K", "J mol-1 K-1", "°C", "Å²"}:
                unit_like_count += 1

        if unit_like_count >= max(2, int(0.25 * len(first))):
            return df.iloc[1:].reset_index(drop=True)

        if "name" in df.columns and str(df.iloc[0].get("name", "")).strip() == "-":
            return df.iloc[1:].reset_index(drop=True)

        return df

    def clean_data(self) -> None:
        if self.raw_df is None:
            raise RuntimeError("load_data() must run first.")

        df = self.raw_df.copy()
        df = df.replace("-", np.nan).replace("—", np.nan).replace("", np.nan)

        for col in [self.cfg.smiles_col, self.cfg.name_col, self.cfg.group_col]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        if self.cfg.target_col not in df.columns:
            raise ValueError(f"Target column '{self.cfg.target_col}' was not found.")
        if self.cfg.smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{self.cfg.smiles_col}' was not found.")

        numeric_cols = [c for c in list(self.cfg.candidate_features) + [self.cfg.target_col] if c in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[self.cfg.target_col]).reset_index(drop=True)

        if self.cfg.iqr_outlier_threshold > 0:
            q1 = df[self.cfg.target_col].quantile(0.25)
            q3 = df[self.cfg.target_col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.cfg.iqr_outlier_threshold * iqr
            high = q3 + self.cfg.iqr_outlier_threshold * iqr
            before = len(df)
            df = df[(df[self.cfg.target_col] >= low) & (df[self.cfg.target_col] <= high)].reset_index(drop=True)
            after = len(df)

            pd.DataFrame([{
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "low_cutoff": low,
                "high_cutoff": high,
                "n_before": before,
                "n_after": after,
                "n_removed": before - after,
            }]).to_csv(self.out_dir / "target_iqr_outlier_audit.csv", index=False)

        if HAS_RDKIT:
            canonical_smiles = []
            valid_rows = []
            for smi in df[self.cfg.smiles_col].astype(str).tolist():
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    valid_rows.append(False)
                    canonical_smiles.append(None)
                else:
                    valid_rows.append(True)
                    canonical_smiles.append(Chem.MolToSmiles(mol))
            df = df.loc[valid_rows].reset_index(drop=True)
            df["canonical_smiles"] = [x for x in canonical_smiles if x is not None]
        else:
            df["canonical_smiles"] = df[self.cfg.smiles_col].astype(str)

        before_dedup = len(df)
        df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
        after_dedup = len(df)

        pd.DataFrame([{
            "n_before_dedup": before_dedup,
            "n_after_dedup": after_dedup,
            "n_removed_duplicates": before_dedup - after_dedup,
            "rdkit_available": HAS_RDKIT,
        }]).to_csv(self.out_dir / "smiles_identity_audit.csv", index=False)

        self.clean_df = df
        self.clean_df.to_csv(self.out_dir / "cleaned_dataset.csv", index=False)

        self._build_feature_quality_summary()
        self._select_retained_features()
        self._build_feature_pairs()

    # ------------------------------------------------------------------
    # feature quality
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
        tmp = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(tmp) < 3:
            return math.nan
        if tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
            return math.nan
        return float(tmp["a"].corr(tmp["b"], method=method))

    def _build_feature_quality_summary(self) -> None:
        if self.clean_df is None:
            raise RuntimeError("clean_df is not ready.")

        df = self.clean_df
        target = df[self.cfg.target_col]
        n = len(df)

        rows = []
        for feat in self.cfg.candidate_features:
            if feat not in df.columns:
                rows.append({
                    "feature": feat,
                    "exists": False,
                    "non_null_count": 0,
                    "non_null_ratio": 0.0,
                    "unique_count": 0,
                    "variance": math.nan,
                    "std": math.nan,
                    "pearson_with_cp": math.nan,
                    "spearman_with_cp": math.nan,
                    "retained": False,
                    "drop_reason": "column_not_found",
                })
                continue

            x = pd.to_numeric(df[feat], errors="coerce")
            non_null_count = int(x.notna().sum())
            non_null_ratio = float(non_null_count / max(n, 1))
            unique_count = int(x.nunique(dropna=True))
            variance = float(x.var(ddof=1)) if non_null_count >= 2 else math.nan
            std = float(x.std(ddof=1)) if non_null_count >= 2 else math.nan

            pearson = self._safe_corr(x, target, "pearson")
            spearman = self._safe_corr(x, target, "spearman")

            retained = True
            drop_reason = "retained"

            if non_null_ratio < self.cfg.min_non_null_ratio:
                retained = False
                drop_reason = "low_non_null_ratio"
            elif unique_count < 2:
                retained = False
                drop_reason = "low_unique_count"
            elif self.cfg.low_variance_threshold > 0 and not math.isnan(variance) and variance <= self.cfg.low_variance_threshold:
                retained = False
                drop_reason = "low_variance"

            rows.append({
                "feature": feat,
                "exists": True,
                "non_null_count": non_null_count,
                "non_null_ratio": non_null_ratio,
                "unique_count": unique_count,
                "variance": variance,
                "std": std,
                "pearson_with_cp": pearson,
                "spearman_with_cp": spearman,
                "abs_pearson_with_cp": abs(pearson) if not math.isnan(pearson) else math.nan,
                "abs_spearman_with_cp": abs(spearman) if not math.isnan(spearman) else math.nan,
                "retained": retained,
                "drop_reason": drop_reason,
            })

        self.feature_quality_df = pd.DataFrame(rows)
        self.feature_quality_df.to_csv(self.out_dir / "feature_quality_summary.csv", index=False, encoding="utf-8-sig")

    def _select_retained_features(self) -> None:
        if self.feature_quality_df is None:
            raise RuntimeError("feature_quality_df is not ready.")

        retained = self.feature_quality_df[self.feature_quality_df["retained"] == True]["feature"].astype(str).tolist()
        if len(retained) < 2:
            raise ValueError("Fewer than two features survived filtering.")
        self.retained_features = retained

    def _build_feature_pairs(self) -> None:
        if self.cfg.fixed_pairs:
            pairs: List[Tuple[str, str]] = []
            for item in self.cfg.fixed_pairs:
                parts = [x.strip() for x in item.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid fixed pair format: {item}")
                if parts[0] == parts[1]:
                    raise ValueError(f"Duplicate feature in fixed pair: {item}")
                pairs.append((parts[0], parts[1]))

            missing = sorted({x for pair in pairs for x in pair if x not in self.retained_features})
            if missing:
                raise ValueError(f"Some fixed-pair features were not retained or not found: {missing}")

            self.feature_pairs = pairs
        else:
            self.feature_pairs = list(combinations(self.retained_features, 2))

    # ------------------------------------------------------------------
    # information gain
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(labels: pd.Series) -> float:
        probs = labels.value_counts(normalize=True)
        if probs.empty:
            return 0.0
        return float(-(probs * np.log2(probs)).sum())

    @staticmethod
    def _discretize_quantile(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()

        out = pd.Series(index=series.index, dtype="object")
        if valid.empty:
            return out

        unique_count = valid.nunique()
        bins = min(n_bins, unique_count)

        if bins < 2:
            out.loc[valid.index] = f"{prefix}_single"
            return out

        ranked = valid.rank(method="first")
        labels = [f"{prefix}_{i}" for i in range(bins)]

        try:
            binned = pd.qcut(ranked, q=bins, labels=labels, duplicates="drop")
            out.loc[valid.index] = binned.astype(str)
        except Exception:
            out.loc[valid.index] = f"{prefix}_failed"

        return out

    def _information_gain(self, x_labels: pd.Series, y_labels: pd.Series) -> float:
        aligned = pd.DataFrame({"x": x_labels, "y": y_labels}).dropna()
        if aligned.empty:
            return math.nan

        hy = self._entropy(aligned["y"])
        conditional_entropy = 0.0
        total = len(aligned)

        for _, group in aligned.groupby("x"):
            weight = len(group) / total
            conditional_entropy += weight * self._entropy(group["y"])

        return float(hy - conditional_entropy)

    def _compute_pair_information_gain(self, pair: Tuple[str, str]) -> Dict[str, Any]:
        if self.clean_df is None:
            raise RuntimeError("clean_df is not ready.")

        f1, f2 = pair
        work = self.clean_df[[f1, f2, self.cfg.target_col]].copy()
        work[f1] = pd.to_numeric(work[f1], errors="coerce")
        work[f2] = pd.to_numeric(work[f2], errors="coerce")
        work[self.cfg.target_col] = pd.to_numeric(work[self.cfg.target_col], errors="coerce")
        work = work.dropna(subset=[f1, f2, self.cfg.target_col]).reset_index(drop=True)

        if len(work) < self.cfg.min_pair_rows:
            return {
                "feature_1": f1,
                "feature_2": f2,
                "ig_n_rows": len(work),
                "ig_feature_1": math.nan,
                "ig_feature_2": math.nan,
                "ig_pair": math.nan,
                "ig_pair_gain_over_best_single": math.nan,
            }

        y_class = self._discretize_quantile(work[self.cfg.target_col], self.cfg.target_bins, "cp_class")
        x1_disc = self._discretize_quantile(work[f1], self.cfg.feature_bins, f1)
        x2_disc = self._discretize_quantile(work[f2], self.cfg.feature_bins, f2)
        x_pair_disc = x1_disc.astype(str) + " | " + x2_disc.astype(str)

        ig1 = self._information_gain(x1_disc, y_class)
        ig2 = self._information_gain(x2_disc, y_class)
        ig_pair = self._information_gain(x_pair_disc, y_class)

        if math.isnan(ig1) or math.isnan(ig2) or math.isnan(ig_pair):
            gain = math.nan
        else:
            gain = ig_pair - max(ig1, ig2)

        return {
            "feature_1": f1,
            "feature_2": f2,
            "ig_n_rows": len(work),
            "ig_feature_1": ig1,
            "ig_feature_2": ig2,
            "ig_pair": ig_pair,
            "ig_pair_gain_over_best_single": gain,
        }

    def compute_information_gain_table(self) -> pd.DataFrame:
        rows = [self._compute_pair_information_gain(pair) for pair in self.feature_pairs]
        self.ig_df = pd.DataFrame(rows)
        self.ig_df.to_csv(self.out_dir / "information_gain_pair_summary.csv", index=False, encoding="utf-8-sig")
        return self.ig_df

    # ------------------------------------------------------------------
    # modeling
    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        out = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        }

        if len(y_true) >= 2 and np.nanstd(y_true) > 0:
            out["r2"] = r2_score(y_true, y_pred)
        else:
            out["r2"] = math.nan

        return out

    def _make_work_df(self, features: Sequence[str]) -> pd.DataFrame:
        if self.clean_df is None:
            raise RuntimeError("clean_df is not ready.")

        cols = [self.cfg.target_col, "canonical_smiles", *features]
        for maybe_col in [self.cfg.name_col, self.cfg.group_col]:
            if maybe_col in self.clean_df.columns:
                cols.append(maybe_col)

        work = self.clean_df[cols].copy()
        for feat in features:
            work[feat] = pd.to_numeric(work[feat], errors="coerce")
        work[self.cfg.target_col] = pd.to_numeric(work[self.cfg.target_col], errors="coerce")
        work = work.dropna(subset=[self.cfg.target_col, *features]).reset_index(drop=True)
        return work

    def _evaluate_cv(self, features: Sequence[str], model_name: str) -> Dict[str, Any]:
        work = self._make_work_df(features)

        if len(work) < max(2, self.cfg.cv_splits):
            return {
                "n_samples_used": len(work),
                "cv_mae_mean": math.nan,
                "cv_mae_std": math.nan,
                "cv_rmse_mean": math.nan,
                "cv_r2_mean": math.nan,
                "skipped": True,
                "skip_reason": "too_few_rows_for_cv",
            }

        X = work[list(features)].values
        y = work[self.cfg.target_col].values.astype(float)

        n_splits = min(self.cfg.cv_splits, len(work))
        if n_splits < 2:
            return {
                "n_samples_used": len(work),
                "cv_mae_mean": math.nan,
                "cv_mae_std": math.nan,
                "cv_rmse_mean": math.nan,
                "cv_r2_mean": math.nan,
                "skipped": True,
                "skip_reason": "n_splits_less_than_2",
            }

        rkf = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=self.cfg.cv_repeats,
            random_state=self.cfg.seed,
        )

        rows = []
        for tr_idx, va_idx in rkf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            pipe = clone(self.pipelines[model_name])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            rows.append(self._metrics(y_va, pred))

        fold_df = pd.DataFrame(rows)
        return {
            "n_samples_used": len(work),
            "cv_mae_mean": float(fold_df["mae"].mean()),
            "cv_mae_std": float(fold_df["mae"].std(ddof=1)),
            "cv_rmse_mean": float(fold_df["rmse"].mean()),
            "cv_r2_mean": float(fold_df["r2"].mean()),
            "skipped": False,
            "skip_reason": "",
        }

    def screen_single_features(self) -> pd.DataFrame:
        rows = []
        total = len(self.retained_features) * len(self.cfg.models)
        idx = 0

        for feat in self.retained_features:
            for model_name in self.cfg.models:
                idx += 1
                print(f"[Single] {idx:3d}/{total:3d} | model={model_name:<10} | feature={feat}")
                result = self._evaluate_cv([feat], model_name)
                rows.append({"feature": feat, "model": model_name, **result})

        df = pd.DataFrame(rows).sort_values(by=["cv_mae_mean", "cv_mae_std"], ascending=[True, True]).reset_index(drop=True)
        self.single_feature_ranking_df = df
        df.to_csv(self.out_dir / "single_feature_ranking.csv", index=False, encoding="utf-8-sig")

        best_single = (
            df[df["skipped"] == False]
            .sort_values(by=["cv_mae_mean", "cv_mae_std"])
            .groupby("feature", as_index=False)
            .first()
        )
        best_single.to_csv(self.out_dir / "single_feature_best_by_feature.csv", index=False, encoding="utf-8-sig")
        return df

    def _get_best_single_mae_map(self) -> Dict[str, float]:
        if self.single_feature_ranking_df is None:
            raise RuntimeError("screen_single_features() must run first.")

        best_single = (
            self.single_feature_ranking_df[self.single_feature_ranking_df["skipped"] == False]
            .sort_values(by=["cv_mae_mean", "cv_mae_std"])
            .groupby("feature", as_index=False)
            .first()
        )
        return dict(zip(best_single["feature"], best_single["cv_mae_mean"]))

    def _pair_diagnostics(self, pair: Tuple[str, str]) -> Dict[str, Any]:
        if self.clean_df is None:
            raise RuntimeError("clean_df is not ready.")

        f1, f2 = pair
        df = self.clean_df

        x1 = pd.to_numeric(df[f1], errors="coerce")
        x2 = pd.to_numeric(df[f2], errors="coerce")
        y = pd.to_numeric(df[self.cfg.target_col], errors="coerce")

        pearson_f1_cp = self._safe_corr(x1, y, "pearson")
        pearson_f2_cp = self._safe_corr(x2, y, "pearson")
        spearman_f1_cp = self._safe_corr(x1, y, "spearman")
        spearman_f2_cp = self._safe_corr(x2, y, "spearman")

        pair_pearson_corr = self._safe_corr(x1, x2, "pearson")
        pair_spearman_corr = self._safe_corr(x1, x2, "spearman")

        if math.isnan(pair_pearson_corr):
            redundancy_flag = False
        else:
            redundancy_flag = abs(pair_pearson_corr) >= self.cfg.redundancy_corr_threshold

        return {
            "pearson_f1_cp": pearson_f1_cp,
            "pearson_f2_cp": pearson_f2_cp,
            "spearman_f1_cp": spearman_f1_cp,
            "spearman_f2_cp": spearman_f2_cp,
            "pair_pearson_corr": pair_pearson_corr,
            "pair_spearman_corr": pair_spearman_corr,
            "pair_redundancy_flag": bool(redundancy_flag),
        }

    def screen_feature_pairs(self) -> pd.DataFrame:
        if self.single_feature_ranking_df is None:
            raise RuntimeError("screen_single_features() must run first.")

        if self.cfg.use_information_gain:
            ig_df = self.compute_information_gain_table()
        else:
            ig_df = pd.DataFrame([{
                "feature_1": f1,
                "feature_2": f2,
                "ig_n_rows": math.nan,
                "ig_feature_1": math.nan,
                "ig_feature_2": math.nan,
                "ig_pair": math.nan,
                "ig_pair_gain_over_best_single": math.nan,
            } for f1, f2 in self.feature_pairs])
            self.ig_df = ig_df

        ig_map = {(row["feature_1"], row["feature_2"]): row.to_dict() for _, row in ig_df.iterrows()}
        best_single_mae = self._get_best_single_mae_map()

        rows = []
        total = len(self.feature_pairs) * len(self.cfg.models)
        idx = 0

        for pair in self.feature_pairs:
            f1, f2 = pair
            pair_rows = len(self._make_work_df([f1, f2]))

            if pair_rows < self.cfg.min_pair_rows:
                for model_name in self.cfg.models:
                    rows.append({
                        "feature_1": f1,
                        "feature_2": f2,
                        "model": model_name,
                        "n_samples_used": pair_rows,
                        "cv_mae_mean": math.nan,
                        "cv_mae_std": math.nan,
                        "cv_rmse_mean": math.nan,
                        "cv_r2_mean": math.nan,
                        "skipped": True,
                        "skip_reason": "too_few_pair_rows",
                        **self._pair_diagnostics(pair),
                        **ig_map.get((f1, f2), {}),
                        "best_single_mae_among_pair_features": math.nan,
                        "mae_improvement_over_best_single": math.nan,
                    })
                continue

            diagnostics = self._pair_diagnostics(pair)
            ig_values = ig_map.get((f1, f2), {})

            best_single_for_pair = min(best_single_mae.get(f1, math.inf), best_single_mae.get(f2, math.inf))
            if math.isinf(best_single_for_pair):
                best_single_for_pair = math.nan

            for model_name in self.cfg.models:
                idx += 1
                print(f"[Pair]   {idx:3d}/{total:3d} | model={model_name:<10} | pair={f1} + {f2}")
                result = self._evaluate_cv([f1, f2], model_name)

                pair_mae = result["cv_mae_mean"]
                if math.isnan(best_single_for_pair) or math.isnan(pair_mae):
                    improvement = math.nan
                else:
                    improvement = best_single_for_pair - pair_mae

                rows.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "model": model_name,
                    **result,
                    **diagnostics,
                    **ig_values,
                    "best_single_mae_among_pair_features": best_single_for_pair,
                    "mae_improvement_over_best_single": improvement,
                })

        df = pd.DataFrame(rows)

        preferred_cols = [
            "feature_1", "feature_2", "model",
            "n_samples_used",
            "cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_r2_mean",
            "best_single_mae_among_pair_features", "mae_improvement_over_best_single",
            "pearson_f1_cp", "pearson_f2_cp", "spearman_f1_cp", "spearman_f2_cp",
            "pair_pearson_corr", "pair_spearman_corr", "pair_redundancy_flag",
            "ig_pair", "ig_pair_gain_over_best_single", "ig_feature_1", "ig_feature_2", "ig_n_rows",
            "skipped", "skip_reason",
        ]
        remaining_cols = [c for c in df.columns if c not in preferred_cols]
        df = df[[c for c in preferred_cols if c in df.columns] + remaining_cols]

        df = df.sort_values(
            by=["cv_mae_mean", "cv_mae_std", "pair_redundancy_flag", "ig_pair_gain_over_best_single"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        self.pair_ranking_df = df
        df.to_csv(self.out_dir / "feature_pair_ranking.csv", index=False, encoding="utf-8-sig")

        best_per_pair = (
            df[df["skipped"] == False]
            .sort_values(
                by=["cv_mae_mean", "cv_mae_std", "pair_redundancy_flag", "ig_pair_gain_over_best_single"],
                ascending=[True, True, True, False],
            )
            .groupby(["feature_1", "feature_2"], as_index=False)
            .first()
            .sort_values(
                by=["cv_mae_mean", "cv_mae_std", "pair_redundancy_flag", "ig_pair_gain_over_best_single"],
                ascending=[True, True, True, False],
            )
            .reset_index(drop=True)
        )

        best_per_pair.to_csv(self.out_dir / "top_unique_feature_pairs.csv", index=False, encoding="utf-8-sig")
        best_per_pair.head(self.cfg.top_k_pairs).to_csv(
            self.out_dir / f"top_{self.cfg.top_k_pairs}_feature_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return df

    # ------------------------------------------------------------------
    # holdout / final selection
    # ------------------------------------------------------------------
    def choose_final_pair_and_model(self) -> Tuple[Tuple[str, str], str]:
        if self.pair_ranking_df is None:
            raise RuntimeError("screen_feature_pairs() must run first.")

        valid = self.pair_ranking_df[self.pair_ranking_df["skipped"] == False].copy()
        if valid.empty:
            raise RuntimeError("No valid pair/model combinations are available.")

        best = valid.iloc[0]
        return (best["feature_1"], best["feature_2"]), str(best["model"])

    def fit_holdout(self, pair: Tuple[str, str], model_name: str) -> HoldoutResult:
        work = self._make_work_df(list(pair))

        y_bins = pd.qcut(
            work[self.cfg.target_col],
            q=min(5, work[self.cfg.target_col].nunique()),
            duplicates="drop",
        )
        stratify = y_bins if y_bins.nunique() >= 2 else None

        idx = np.arange(len(work))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=self.cfg.test_size,
            random_state=self.cfg.seed,
            stratify=stratify,
        )

        train_df = work.iloc[train_idx].copy().reset_index(drop=True)
        test_df = work.iloc[test_idx].copy().reset_index(drop=True)

        pipe = clone(self.pipelines[model_name])
        pipe.fit(train_df[list(pair)].values, train_df[self.cfg.target_col].values.astype(float))

        pred_train = pipe.predict(train_df[list(pair)].values)
        pred_test = pipe.predict(test_df[list(pair)].values)

        train_df["y_pred"] = pred_train
        train_df["abs_error"] = np.abs(train_df[self.cfg.target_col] - train_df["y_pred"])
        train_df["residual"] = train_df[self.cfg.target_col] - train_df["y_pred"]

        test_df["y_pred"] = pred_test
        test_df["abs_error"] = np.abs(test_df[self.cfg.target_col] - test_df["y_pred"])
        test_df["residual"] = test_df[self.cfg.target_col] - test_df["y_pred"]

        train_metrics = self._metrics(train_df[self.cfg.target_col].values, pred_train)
        test_metrics = self._metrics(test_df[self.cfg.target_col].values, pred_test)

        return HoldoutResult(
            model_name=model_name,
            feature_pair=pair,
            target_col=self.cfg.target_col,
            mae_train=train_metrics["mae"],
            mae_test=test_metrics["mae"],
            rmse_test=test_metrics["rmse"],
            r2_test=test_metrics["r2"],
            fitted_pipeline=pipe,
            train_df=train_df,
            test_df=test_df,
        )

    # ------------------------------------------------------------------
    # explanation / shap
    # ------------------------------------------------------------------
    def explain(self, result: HoldoutResult) -> pd.DataFrame:
        feature_names = list(result.feature_pair)
        X_test_raw = result.test_df[feature_names].copy()
        y_test = result.test_df[result.target_col].values

        perm = permutation_importance(
            result.fitted_pipeline,
            X_test_raw.values,
            y_test,
            n_repeats=30,
            random_state=self.cfg.seed,
            scoring="neg_mean_absolute_error",
        )

        perm_df = pd.DataFrame({
            "feature": feature_names,
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
            "explanation_type": "permutation_importance",
        })

        if self.cfg.no_shap or (not HAS_SHAP):
            return perm_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

        try:
            model = result.fitted_pipeline.named_steps["model"]
            prep = result.fitted_pipeline.named_steps["prep"]
            X_train_raw = result.train_df[feature_names].copy()
            X_train = prep.transform(X_train_raw.values)
            X_test = prep.transform(X_test_raw.values)

            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
                explainer = shap.TreeExplainer(model)
                values = np.asarray(explainer.shap_values(X_test))
                explanation_type = "shap_tree_mean_abs"
            elif isinstance(model, (LinearRegression, Ridge, ElasticNet)):
                try:
                    explainer = shap.LinearExplainer(model, X_train)
                    values = np.asarray(explainer.shap_values(X_test))
                except Exception:
                    explainer = shap.Explainer(model, X_train)
                    values = np.asarray(explainer(X_test).values)
                explanation_type = "shap_linear_mean_abs"
            else:
                background = shap.sample(pd.DataFrame(X_train, columns=feature_names), min(50, len(X_train)))
                explainer = shap.KernelExplainer(lambda z: model.predict(np.asarray(z)), background)
                values = np.asarray(
                    explainer.shap_values(
                        pd.DataFrame(X_test, columns=feature_names),
                        nsamples=min(100, 2 * len(X_test) + 10),
                    )
                )
                explanation_type = "shap_kernel_mean_abs"

            if values.ndim == 1:
                values = values.reshape(-1, 1)

            shap_df = pd.DataFrame({
                "feature": feature_names,
                "importance": np.mean(np.abs(values), axis=0),
                "importance_std": np.std(np.abs(values), axis=0, ddof=1) if len(values) > 1 else np.zeros(len(feature_names)),
                "explanation_type": explanation_type,
            })
            return shap_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        except Exception:
            return perm_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    def _compute_shap_payload(self, result: HoldoutResult) -> Tuple[Optional[dict], str]:
        if self.cfg.no_shap or (not HAS_SHAP):
            return None, "shap_disabled_or_unavailable"

        feature_names = list(result.feature_pair)
        model = result.fitted_pipeline.named_steps["model"]
        prep = result.fitted_pipeline.named_steps["prep"]

        X_train_raw = result.train_df[feature_names].copy()
        X_test_raw = result.test_df[feature_names].copy()

        X_train = prep.transform(X_train_raw.values)
        X_test = prep.transform(X_test_raw.values)

        try:
            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
                explainer = shap.TreeExplainer(model)
                values = np.asarray(explainer.shap_values(X_test))
                explanation_type = "shap_tree"
            elif isinstance(model, (LinearRegression, Ridge, ElasticNet)):
                try:
                    explainer = shap.LinearExplainer(model, X_train)
                    values = np.asarray(explainer.shap_values(X_test))
                except Exception:
                    explainer = shap.Explainer(model, X_train)
                    values = np.asarray(explainer(X_test).values)
                explanation_type = "shap_linear"
            else:
                background = shap.sample(pd.DataFrame(X_train, columns=feature_names), min(50, len(X_train)))
                explain_df = pd.DataFrame(X_test, columns=feature_names)
                explainer = shap.KernelExplainer(lambda z: model.predict(np.asarray(z)), background)
                values = np.asarray(
                    explainer.shap_values(
                        explain_df,
                        nsamples=min(100, 2 * len(explain_df) + 10),
                    )
                )
                explanation_type = "shap_kernel"

            if values.ndim == 1:
                values = values.reshape(-1, 1)

            payload = {
                "values": values,
                "X_raw": X_test_raw.reset_index(drop=True),
                "feature_names": feature_names,
            }
            return payload, explanation_type
        except Exception:
            return None, "shap_failed"

    def _save_shap_beeswarm_plot(self, shap_payload: dict, out_png: Path, title: str) -> None:
        values = shap_payload["values"]
        X_raw = shap_payload["X_raw"]
        feature_names = shap_payload["feature_names"]

        plt.figure(figsize=(7.5, max(3.8, 0.85 * len(feature_names) + 2.0)))
        shap.summary_plot(
            values,
            features=X_raw,
            feature_names=feature_names,
            show=False,
            max_display=len(feature_names),
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_shap_bar_plot(self, shap_payload: dict, out_png: Path, title: str) -> None:
        values = shap_payload["values"]
        X_raw = shap_payload["X_raw"]
        feature_names = shap_payload["feature_names"]

        plt.figure(figsize=(7.5, max(3.8, 0.85 * len(feature_names) + 1.5)))
        shap.summary_plot(
            values,
            features=X_raw,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=len(feature_names),
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_shap_dependence_plots(self, shap_payload: dict, out_prefix: Path, title_prefix: str) -> None:
        values = shap_payload["values"]
        X_raw = shap_payload["X_raw"]
        feature_names = shap_payload["feature_names"]

        for feat in feature_names:
            plt.figure(figsize=(6.5, 5.0))
            shap.dependence_plot(
                feat,
                values,
                X_raw,
                feature_names=feature_names,
                show=False,
                interaction_index="auto",
            )
            plt.title(f"{title_prefix} | {feat}")
            plt.tight_layout()
            safe_feat = str(feat).replace("/", "_")
            plt.savefig(
                out_prefix.parent / f"{out_prefix.name}__dep__{safe_feat}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # ------------------------------------------------------------------
    # plot helpers
    # ------------------------------------------------------------------
    def _classify_compound_type(self, df: pd.DataFrame) -> pd.Series:
        if self.cfg.group_col in df.columns:
            raw = df[self.cfg.group_col].astype(str).str.strip()
            raw = raw.replace({"nan": "Unknown", "": "Unknown"})
            return raw

        smiles = df[self.cfg.smiles_col].astype(str)

        labels = []
        for smi in smiles:
            if "N" in smi:
                labels.append("Nitrogen-containing")
            elif "S" in smi:
                labels.append("Sulfur-containing")
            elif any(x in smi for x in ["Cl", "Br", "F", "I"]):
                labels.append("Halides")
            elif "O" in smi:
                labels.append("Oxygenated compounds")
            else:
                labels.append("Hydrocarbons")
        return pd.Series(labels, index=df.index)

    def _save_cp_distribution_histogram(self) -> None:
        if self.clean_df is None:
            return

        y = pd.to_numeric(self.clean_df[self.cfg.target_col], errors="coerce").dropna()
        if y.empty:
            return

        plt.figure(figsize=(7.2, 5.2))
        plt.hist(y, bins=min(30, max(10, int(math.sqrt(len(y))))))

        plt.xlabel(self.cfg.target_col)
        plt.ylabel("Count")
        plt.title("Cp distribution histogram")
        plt.grid(True, axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "cp_distribution_histogram.png", dpi=180)
        plt.close()

    def _save_cp_distribution_violin(self) -> None:
        if self.clean_df is None:
            return

        df = self.clean_df.copy()
        df["compound_type_plot"] = self._classify_compound_type(df)
        df[self.cfg.target_col] = pd.to_numeric(df[self.cfg.target_col], errors="coerce")
        df = df.dropna(subset=[self.cfg.target_col])

        if df.empty:
            return

        order = df["compound_type_plot"].value_counts().index.tolist()
        groups = [df.loc[df["compound_type_plot"] == g, self.cfg.target_col].values for g in order]
        groups = [g for g in groups if len(g) > 0]
        labels = [g for g in order if len(df.loc[df["compound_type_plot"] == g]) > 0]

        if len(groups) < 1:
            return

        plt.figure(figsize=(10, 5.8))
        parts = plt.violinplot(groups, showmeans=False, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)

        plt.xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
        plt.ylabel(self.cfg.target_col)
        plt.title("Cp distribution by compound type")
        plt.grid(True, axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "cp_distribution_by_compound_type_violin.png", dpi=180)
        plt.close()

    def _save_feature_correlation_heatmap(self) -> None:
        if self.clean_df is None:
            return

        cols = [self.cfg.target_col] + self.retained_features
        corr_df = self.clean_df[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
        corr_df.to_csv(self.out_dir / "feature_target_correlation_matrix.csv", encoding="utf-8-sig")

        labels = corr_df.columns.tolist()
        mat = corr_df.values

        fig_size = max(6.0, 0.65 * len(labels))
        plt.figure(figsize=(fig_size, fig_size))
        im = plt.imshow(mat, vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson correlation")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = mat[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

        plt.title("Feature / Cp Pearson correlation heatmap")
        plt.tight_layout()
        plt.savefig(self.out_dir / "feature_target_correlation_heatmap.png", dpi=180)
        plt.close()

    def _save_top_feature_pairs_mae_plot(self) -> None:
        if self.pair_ranking_df is None or self.pair_ranking_df.empty:
            return

        top = self.pair_ranking_df[self.pair_ranking_df["skipped"] == False].head(self.cfg.top_n_plot).copy()
        if top.empty:
            return

        labels = top.apply(lambda r: f"{r['model']} | {r['feature_1']} + {r['feature_2']}", axis=1).tolist()
        y_pos = np.arange(len(top))[::-1]

        plt.figure(figsize=(12, max(4.5, 0.45 * len(top))))
        plt.barh(y_pos, top["cv_mae_mean"].values, xerr=top["cv_mae_std"].values)
        plt.yticks(y_pos, labels)
        plt.xlabel("Repeated CV MAE")
        plt.title("Top feature-pair regression results")

        for y, (_, row) in zip(y_pos, top.iterrows()):
            ig = row.get("ig_pair", math.nan)
            gain = row.get("ig_pair_gain_over_best_single", math.nan)
            red = row.get("pair_redundancy_flag", False)
            text = f"IG={ig:.3f}, ΔIG={gain:.3f}, red={red}"
            plt.text(row["cv_mae_mean"], y, "  " + text, va="center", fontsize=8)

        plt.grid(True, axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "top_feature_pairs_mae.png", dpi=180)
        plt.close()

    def _save_information_gain_top_pairs_plot(self) -> None:
        if self.ig_df is None or self.ig_df.empty:
            return

        top = self.ig_df.sort_values(by="ig_pair", ascending=False).head(self.cfg.top_n_plot).copy()
        if top.empty:
            return

        labels = top.apply(lambda r: f"{r['feature_1']} + {r['feature_2']}", axis=1).tolist()
        y_pos = np.arange(len(top))[::-1]

        plt.figure(figsize=(10.5, max(4.5, 0.4 * len(top))))
        plt.barh(y_pos, top["ig_pair"].values)
        plt.yticks(y_pos, labels)
        plt.xlabel("Information Gain")
        plt.title("Top feature pairs by Information Gain")
        plt.grid(True, axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "information_gain_top_pairs.png", dpi=180)
        plt.close()

    def _save_ig_vs_mae_plot(self) -> None:
        if self.pair_ranking_df is None or self.pair_ranking_df.empty:
            return

        df = self.pair_ranking_df[
            (self.pair_ranking_df["skipped"] == False)
            & self.pair_ranking_df["ig_pair"].notna()
            & self.pair_ranking_df["cv_mae_mean"].notna()
        ].copy()

        if df.empty:
            return

        plt.figure(figsize=(7.2, 5.8))
        plt.scatter(df["ig_pair"], df["cv_mae_mean"], alpha=0.75)

        top = df.sort_values(by="cv_mae_mean", ascending=True).head(8)
        for _, row in top.iterrows():
            label = f"{row['feature_1']}+{row['feature_2']}\n{row['model']}"
            plt.annotate(label, (row["ig_pair"], row["cv_mae_mean"]), fontsize=8, xytext=(5, 5), textcoords="offset points")

        plt.xlabel("Information Gain of feature pair")
        plt.ylabel("Repeated CV MAE")
        plt.title("Information Gain vs regression error")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "information_gain_vs_cv_mae.png", dpi=180)
        plt.close()

    def _save_ig_pair_heatmap(self) -> None:
        if self.ig_df is None or self.ig_df.empty:
            return

        features = self.retained_features
        mat = pd.DataFrame(np.nan, index=features, columns=features)

        for _, row in self.ig_df.iterrows():
            f1 = row["feature_1"]
            f2 = row["feature_2"]
            value = row["ig_pair"]
            if f1 in mat.index and f2 in mat.columns:
                mat.loc[f1, f2] = value
                mat.loc[f2, f1] = value

        mat.to_csv(self.out_dir / "information_gain_pair_matrix.csv", encoding="utf-8-sig")

        fig_size = max(6.0, 0.65 * len(features))
        plt.figure(figsize=(fig_size, fig_size))
        im = plt.imshow(mat.values)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Pair Information Gain")
        plt.xticks(range(len(features)), features, rotation=45, ha="right")
        plt.yticks(range(len(features)), features)

        for i in range(len(features)):
            for j in range(len(features)):
                value = mat.values[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

        plt.title("Pairwise Information Gain heatmap")
        plt.tight_layout()
        plt.savefig(self.out_dir / "information_gain_pair_heatmap.png", dpi=180)
        plt.close()

    def _save_feature_count_performance_curve(self) -> None:
        if self.single_feature_ranking_df is None or self.pair_ranking_df is None:
            return

        single_valid = self.single_feature_ranking_df[self.single_feature_ranking_df["skipped"] == False].copy()
        pair_valid = self.pair_ranking_df[self.pair_ranking_df["skipped"] == False].copy()

        if single_valid.empty or pair_valid.empty:
            return

        best_single = single_valid.sort_values(by="cv_mae_mean").iloc[0]
        best_pair = pair_valid.sort_values(by="cv_mae_mean").iloc[0]

        summary = pd.DataFrame([
            {
                "n_features": 1,
                "label": f"{best_single['feature']} | {best_single['model']}",
                "best_cv_mae": best_single["cv_mae_mean"],
                "cv_mae_std": best_single["cv_mae_std"],
            },
            {
                "n_features": 2,
                "label": f"{best_pair['feature_1']} + {best_pair['feature_2']} | {best_pair['model']}",
                "best_cv_mae": best_pair["cv_mae_mean"],
                "cv_mae_std": best_pair["cv_mae_std"],
            },
        ])

        summary.to_csv(self.out_dir / "feature_count_performance_summary.csv", index=False, encoding="utf-8-sig")

        plt.figure(figsize=(6.0, 4.5))
        plt.errorbar(summary["n_features"], summary["best_cv_mae"], yerr=summary["cv_mae_std"], marker="o", capsize=4)
        plt.xticks([1, 2])
        plt.xlabel("Number of physical features")
        plt.ylabel("Best repeated CV MAE")
        plt.title("Best 1-feature vs 2-feature performance")
        plt.grid(True, linestyle=":", alpha=0.5)

        for _, row in summary.iterrows():
            plt.annotate(row["label"], (row["n_features"], row["best_cv_mae"]), xytext=(5, 6), textcoords="offset points", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.out_dir / "feature_count_performance_curve.png", dpi=180)
        plt.close()

    def _save_holdout_parity_plot(self, result: HoldoutResult) -> None:
        y_tr = result.train_df[result.target_col].values
        yp_tr = result.train_df["y_pred"].values
        y_te = result.test_df[result.target_col].values
        yp_te = result.test_df["y_pred"].values

        all_vals = np.concatenate([y_tr, yp_tr, y_te, yp_te])
        lo = float(np.min(all_vals) - 5)
        hi = float(np.max(all_vals) + 5)

        plt.figure(figsize=(6.5, 6.0))
        plt.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect")
        plt.scatter(y_tr, yp_tr, alpha=0.35, s=18, label=f"Train (MAE={result.mae_train:.2f})")
        plt.scatter(y_te, yp_te, alpha=0.85, s=35, marker="s", label=f"Test (MAE={result.mae_test:.2f})")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel(f"Actual {result.target_col}")
        plt.ylabel(f"Predicted {result.target_col}")
        plt.title(f"Best holdout model\n{result.model_name} | {result.feature_pair[0]} + {result.feature_pair[1]}")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(self.out_dir / "best_model_holdout_parity.png", dpi=180)
        plt.close()

    def _save_holdout_residual_plot(self, result: HoldoutResult) -> None:
        tr = result.train_df.copy()
        te = result.test_df.copy()

        plt.figure(figsize=(7.0, 5.4))
        plt.axhline(0.0, linestyle="--", linewidth=1.2)
        plt.scatter(tr["y_pred"], tr["residual"], alpha=0.35, s=18, label="Train")
        plt.scatter(te["y_pred"], te["residual"], alpha=0.85, s=35, marker="s", label="Test")

        plt.xlabel("Predicted value")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title(f"Residual plot\n{result.model_name} | {result.feature_pair[0]} + {result.feature_pair[1]}")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "best_model_residual_plot.png", dpi=180)
        plt.close()

    def _save_explanation_bar(self, df: pd.DataFrame, out_png: Path, title: str) -> None:
        plot_df = df.sort_values(by="importance", ascending=True)
        plt.figure(figsize=(7, max(3.5, 0.6 * len(plot_df))))
        plt.barh(plot_df["feature"], plot_df["importance"].values, xerr=plot_df["importance_std"].values)
        plt.xlabel("Mean absolute contribution / importance")
        plt.title(title)
        plt.grid(True, axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()

    # ------------------------------------------------------------------
    # save outputs
    # ------------------------------------------------------------------
    def save_holdout_outputs(self, result: HoldoutResult) -> None:
        result.train_df.to_csv(self.out_dir / "holdout_predictions_train_best_model.csv", index=False)
        result.test_df.to_csv(self.out_dir / "holdout_predictions_best_model.csv", index=False)

        cols = [c for c in [
            self.cfg.name_col,
            "canonical_smiles",
            *result.feature_pair,
            result.target_col,
            "y_pred",
            "abs_error",
            "residual",
        ] if c in result.test_df.columns]

        worst = result.test_df.sort_values(by="abs_error", ascending=False).head(15)[cols]
        worst.to_csv(self.out_dir / "worst_predictions_best_model.csv", index=False)

        pd.DataFrame([{
            "model": result.model_name,
            "feature_1": result.feature_pair[0],
            "feature_2": result.feature_pair[1],
            "mae_train": result.mae_train,
            "mae_test": result.mae_test,
            "rmse_test": result.rmse_test,
            "r2_test": result.r2_test,
            "n_train": len(result.train_df),
            "n_test": len(result.test_df),
        }]).to_csv(self.out_dir / "best_model_holdout_summary.csv", index=False)

        if self.cfg.save_holdout_plots:
            self._save_holdout_parity_plot(result)
            self._save_holdout_residual_plot(result)

        imp_df = self.explain(result)
        imp_df.to_csv(self.out_dir / "best_model_explanation.csv", index=False)
        self._save_explanation_bar(
            imp_df,
            self.out_dir / "best_model_explanation.png",
            f"Best model explanation | {result.model_name} | {result.feature_pair[0]} + {result.feature_pair[1]}",
        )

        shap_payload, shap_plot_type = self._compute_shap_payload(result)
        if shap_payload is not None:
            if self.cfg.save_shap_beeswarm:
                self._save_shap_beeswarm_plot(
                    shap_payload,
                    self.out_dir / "best_model_shap_beeswarm.png",
                    f"Best model SHAP beeswarm | {result.model_name} | {result.feature_pair[0]} + {result.feature_pair[1]}",
                )
                self._save_shap_bar_plot(
                    shap_payload,
                    self.out_dir / "best_model_shap_bar.png",
                    f"Best model SHAP bar | {result.model_name} | {result.feature_pair[0]} + {result.feature_pair[1]}",
                )

            if self.cfg.save_shap_dependence:
                self._save_shap_dependence_plots(
                    shap_payload,
                    self.out_dir / "best_model",
                    f"Best model | {result.model_name}",
                )

    def save_run_config(self) -> None:
        config_dict = asdict(self.cfg)

        with open(self.out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        if HAS_YAML:
            with open(self.out_dir / "run_config_used.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)

    def save_final_summary(self, result: HoldoutResult) -> None:
        if self.pair_ranking_df is None:
            return

        valid = self.pair_ranking_df[self.pair_ranking_df["skipped"] == False].copy()
        if valid.empty:
            return

        best = valid.iloc[0].to_dict()

        summary = {
            "best_feature_1": best.get("feature_1"),
            "best_feature_2": best.get("feature_2"),
            "best_model_cv": best.get("model"),
            "best_cv_mae_mean": best.get("cv_mae_mean"),
            "best_cv_mae_std": best.get("cv_mae_std"),
            "best_cv_rmse_mean": best.get("cv_rmse_mean"),
            "best_cv_r2_mean": best.get("cv_r2_mean"),
            "best_ig_pair": best.get("ig_pair"),
            "best_ig_pair_gain_over_best_single": best.get("ig_pair_gain_over_best_single"),
            "best_pair_pearson_corr": best.get("pair_pearson_corr"),
            "best_pair_redundancy_flag": best.get("pair_redundancy_flag"),
            "best_single_mae_among_pair_features": best.get("best_single_mae_among_pair_features"),
            "mae_improvement_over_best_single": best.get("mae_improvement_over_best_single"),
            "holdout_model": result.model_name,
            "holdout_pair_feature_1": result.feature_pair[0],
            "holdout_pair_feature_2": result.feature_pair[1],
            "holdout_mae_train": result.mae_train,
            "holdout_mae_test": result.mae_test,
            "holdout_rmse_test": result.rmse_test,
            "holdout_r2_test": result.r2_test,
        }

        pd.DataFrame([summary]).to_csv(self.out_dir / "best_feature_pair_summary.csv", index=False, encoding="utf-8-sig")

    def save_all_plots(self) -> None:
        if not self.cfg.save_plots:
            return

        if self.cfg.save_distribution_plots:
            self._save_cp_distribution_histogram()

        if self.cfg.save_violin_plot:
            self._save_cp_distribution_violin()

        if self.cfg.save_correlation_heatmap:
            self._save_feature_correlation_heatmap()

        self._save_top_feature_pairs_mae_plot()
        self._save_feature_count_performance_curve()

        if self.cfg.save_ig_plots:
            self._save_information_gain_top_pairs_plot()
            self._save_ig_vs_mae_plot()
            self._save_ig_pair_heatmap()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self) -> HoldoutResult:
        print("=" * 90)
        print("Cp two-feature selector with holdout / SHAP outputs")
        print("=" * 90)
        print(f"Config file       : {self.cfg.config_file}")
        print(f"Data file         : {self.data_path}")
        print(f"Output dir        : {self.out_dir}")
        print(f"Target column     : {self.cfg.target_col}")
        print(f"Candidate features: {list(self.cfg.candidate_features)}")
        print(f"Models            : {list(self.cfg.models)}")
        print(f"Random seed       : {self.cfg.seed}")
        print()

        self.load_data()
        self.clean_data()

        assert self.raw_df is not None
        assert self.clean_df is not None

        print(f"[1] Raw shape              : {self.raw_df.shape}")
        print(f"[1] Clean shape            : {self.clean_df.shape}")
        print(f"[1] Retained features      : {self.retained_features}")
        print(f"[1] Number of feature pairs: {len(self.feature_pairs)}")
        print()

        print("[2] Screening single features...")
        single_df = self.screen_single_features()
        print()
        print(single_df.head(10).to_string(index=False))
        print()

        print("[3] Screening feature pairs...")
        pair_df = self.screen_feature_pairs()
        print()
        print(pair_df.head(15).to_string(index=False))
        print()

        print("[4] Saving selection-stage plots...")
        self.save_all_plots()

        print("[5] Fitting final holdout model...")
        final_pair, final_model = self.choose_final_pair_and_model()
        print(f"Final holdout choice: model={final_model}, pair={final_pair[0]} + {final_pair[1]}")
        result = self.fit_holdout(final_pair, final_model)
        self.save_holdout_outputs(result)

        print("[6] Saving final summaries...")
        self.save_final_summary(result)
        self.save_run_config()

        print()
        print("=" * 90)
        print("Finished.")
        print("=" * 90)
        print(f"Best feature pair (CV) : {final_pair[0]} + {final_pair[1]}")
        print(f"Holdout model          : {result.model_name}")
        print(f"Holdout test MAE       : {result.mae_test:.4f}")
        print(f"Holdout test RMSE      : {result.rmse_test:.4f}")
        print(f"Holdout test R2        : {result.r2_test:.4f}")
        print(f"All outputs saved to   : {self.out_dir}")
        return result


# ----------------------------------------------------------------------
# cli / config
# ----------------------------------------------------------------------
def load_yaml_config(path: str) -> Dict[str, Any]:
    if not HAS_YAML:
        raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML top level must be a mapping/dictionary.")

    return data


def _validate_model_names(models: Sequence[str]) -> Tuple[str, ...]:
    invalid = [m for m in models if m not in DEFAULT_MODELS]
    if invalid:
        raise ValueError(f"Invalid model names: {invalid}. Allowed: {DEFAULT_MODELS}")
    return tuple(models)


def parse_args() -> FeatureSelectorConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Cp feature-pair selector with holdout / SHAP outputs.",
        parents=[pre_parser],
    )

    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--smiles-col", type=str, default=None)
    parser.add_argument("--name-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--group-col", type=str, default=None)

    parser.add_argument("--candidate-features", nargs="*", default=None)
    parser.add_argument("--fixed-pairs", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)

    parser.add_argument("--min-non-null-ratio", type=float, default=None)
    parser.add_argument("--min-pair-rows", type=int, default=None)
    parser.add_argument("--low-variance-threshold", type=float, default=None)
    parser.add_argument("--iqr-outlier-threshold", type=float, default=None)
    parser.add_argument("--redundancy-corr-threshold", type=float, default=None)

    parser.add_argument("--target-bins", type=int, default=None)
    parser.add_argument("--feature-bins", type=int, default=None)

    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k-pairs", type=int, default=None)
    parser.add_argument("--top-n-plot", type=int, default=None)

    parser.add_argument("--skip-unit-row", dest="skip_unit_row", action="store_true")
    parser.add_argument("--no-skip-unit-row", dest="skip_unit_row", action="store_false")
    parser.set_defaults(skip_unit_row=None)

    parser.add_argument("--auto-drop-unit-row", dest="auto_drop_unit_row", action="store_true")
    parser.add_argument("--no-auto-drop-unit-row", dest="auto_drop_unit_row", action="store_false")
    parser.set_defaults(auto_drop_unit_row=None)

    parser.add_argument("--use-information-gain", dest="use_information_gain", action="store_true")
    parser.add_argument("--no-information-gain", dest="use_information_gain", action="store_false")
    parser.set_defaults(use_information_gain=None)

    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=None)

    parser.add_argument("--save-distribution-plots", dest="save_distribution_plots", action="store_true")
    parser.add_argument("--no-distribution-plots", dest="save_distribution_plots", action="store_false")
    parser.set_defaults(save_distribution_plots=None)

    parser.add_argument("--save-violin-plot", dest="save_violin_plot", action="store_true")
    parser.add_argument("--no-violin-plot", dest="save_violin_plot", action="store_false")
    parser.set_defaults(save_violin_plot=None)

    parser.add_argument("--save-correlation-heatmap", dest="save_correlation_heatmap", action="store_true")
    parser.add_argument("--no-correlation-heatmap", dest="save_correlation_heatmap", action="store_false")
    parser.set_defaults(save_correlation_heatmap=None)

    parser.add_argument("--save-ig-plots", dest="save_ig_plots", action="store_true")
    parser.add_argument("--no-ig-plots", dest="save_ig_plots", action="store_false")
    parser.set_defaults(save_ig_plots=None)

    parser.add_argument("--save-holdout-plots", dest="save_holdout_plots", action="store_true")
    parser.add_argument("--no-holdout-plots", dest="save_holdout_plots", action="store_false")
    parser.set_defaults(save_holdout_plots=None)

    parser.add_argument("--no-shap", dest="no_shap", action="store_true")
    parser.add_argument("--use-shap", dest="no_shap", action="store_false")
    parser.set_defaults(no_shap=None)

    parser.add_argument("--save-shap-beeswarm", dest="save_shap_beeswarm", action="store_true")
    parser.add_argument("--no-save-shap-beeswarm", dest="save_shap_beeswarm", action="store_false")
    parser.set_defaults(save_shap_beeswarm=None)

    parser.add_argument("--save-shap-dependence", dest="save_shap_dependence", action="store_true")
    parser.add_argument("--no-save-shap-dependence", dest="save_shap_dependence", action="store_false")
    parser.set_defaults(save_shap_dependence=None)

    args = parser.parse_args(remaining)

    config = FeatureSelectorConfig(config_file=pre_args.config)

    yaml_data: Dict[str, Any] = {}
    if pre_args.config is not None:
        yaml_data = load_yaml_config(pre_args.config)

    valid_fields = {f.name for f in fields(FeatureSelectorConfig)}
    unknown_yaml_keys = [k for k in yaml_data if k not in valid_fields]
    if unknown_yaml_keys:
        raise ValueError(f"Unknown YAML keys: {unknown_yaml_keys}")

    for f in fields(FeatureSelectorConfig):
        if f.name in yaml_data:
            setattr(config, f.name, yaml_data[f.name])

    for f in fields(FeatureSelectorConfig):
        if f.name == "config_file":
            continue
        if hasattr(args, f.name):
            value = getattr(args, f.name)
            if value is not None:
                setattr(config, f.name, value)

    config.candidate_features = tuple(config.candidate_features)
    config.models = _validate_model_names(tuple(config.models))

    if config.fixed_pairs is not None:
        config.fixed_pairs = list(config.fixed_pairs)

    return config


def main() -> None:
    config = parse_args()
    selector = CpFeaturePairSelector(config)
    selector.run()


if __name__ == "__main__":
    main()
