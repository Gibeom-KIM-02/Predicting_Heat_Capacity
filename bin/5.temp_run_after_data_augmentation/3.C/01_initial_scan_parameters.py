#!/usr/bin/env python3
"""
Lean class-based Cp prediction experiment framework.
YAML-first version with CLI override support.

Typical usage
-------------
python 01_initial_scan_parameters.py --config config.yaml
python 01_initial_scan_parameters.py --config config.yaml --seed 777 --output-dir RESULTS_TRY2
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
class ExperimentConfig:
    config_file: Optional[str] = None
    data_file: str = "mid_project_verified.xlsx"
    sheet_name: Optional[str] = None
    smiles_col: str = "smiles"
    name_col: str = "name"
    target_col: str = "heat_capacity"
    candidate_features: Sequence[str] = tuple(DEFAULT_CANDIDATE_FEATURES)
    fixed_pairs: Optional[Sequence[str]] = None
    models: Sequence[str] = tuple(DEFAULT_MODELS)
    top_k_pairs: int = 5
    cv_splits: int = 5
    cv_repeats: int = 10
    test_size: float = 0.2
    seed: int = 2357
    output_dir: str = "RESULTS"
    min_non_null_ratio: float = 0.7
    iqr_outlier_threshold: float = 0.0
    explain_top_pairs: bool = False
    top_pairs_to_explain: int = 5
    no_shap: bool = False
    save_shap_beeswarm: bool = False
    save_shap_dependence: bool = False
    holdout_model: Optional[str] = None
    holdout_pair: Optional[str] = None
    model_params: Optional[Dict[str, Dict[str, Any]]] = None


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


class CpExperiment:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.data_path = Path(config.data_file).resolve()
        self.out_dir = Path(config.output_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.retained_features: List[str] = []
        self.feature_pairs: List[Tuple[str, str]] = []
        self.pipelines: Dict[str, Pipeline] = self._build_model_pipelines()
        self.ranking_df: Optional[pd.DataFrame] = None

        self._set_seed()

    # ------------------------------------------------------------------
    # Core setup
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
            "elasticnet": Pipeline([
                ("prep", scaled),
                ("model", ElasticNet(**elasticnet_params)),
            ]),
            "svr": Pipeline([("prep", scaled), ("model", SVR(**svr_params))]),
            "rf": Pipeline([
                ("prep", unscaled),
                ("model", RandomForestRegressor(**rf_params)),
            ]),
            "gbr": Pipeline([
                ("prep", unscaled),
                ("model", GradientBoostingRegressor(**gbr_params)),
            ]),
        }

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        if self.cfg.sheet_name is None:
            self.raw_df = pd.read_excel(self.data_path, skiprows=[1])
        else:
            self.raw_df = pd.read_excel(self.data_path, sheet_name=self.cfg.sheet_name, skiprows=[1])

    def clean_data(self) -> None:
        if self.raw_df is None:
            raise RuntimeError("load_data() must run first.")

        df = self.raw_df.copy()
        df = df.replace("-", np.nan).replace("—", np.nan)

        for col in [self.cfg.smiles_col, self.cfg.name_col]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        numeric_cols = [
            c for c in list(self.cfg.candidate_features) + [self.cfg.target_col] if c in df.columns
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.cfg.target_col not in df.columns:
            raise ValueError(f"Target column '{self.cfg.target_col}' not found.")
        if self.cfg.smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{self.cfg.smiles_col}' not found.")

        df = df.dropna(subset=[self.cfg.target_col]).reset_index(drop=True)

        if self.cfg.iqr_outlier_threshold > 0:
            q1 = df[self.cfg.target_col].quantile(0.25)
            q3 = df[self.cfg.target_col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.cfg.iqr_outlier_threshold * iqr
            high = q3 + self.cfg.iqr_outlier_threshold * iqr
            df = df[(df[self.cfg.target_col] >= low) & (df[self.cfg.target_col] <= high)].reset_index(drop=True)

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

        df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

        retained = []
        n = len(df)
        for feat in self.cfg.candidate_features:
            if feat not in df.columns:
                continue
            ratio = df[feat].notna().sum() / max(n, 1)
            if ratio >= self.cfg.min_non_null_ratio:
                retained.append(feat)

        if len(retained) < 2:
            raise ValueError("Fewer than two candidate features survived filtering.")

        self.clean_df = df
        self.retained_features = retained
        self.feature_pairs = self._build_feature_pairs()

        df.to_csv(self.out_dir / "cleaned_dataset.csv", index=False)
        self._save_feature_availability()

    def _build_feature_pairs(self) -> List[Tuple[str, str]]:
        if self.cfg.fixed_pairs:
            pairs: List[Tuple[str, str]] = []
            for item in self.cfg.fixed_pairs:
                parts = [x.strip() for x in item.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid pair format: {item}")
                if parts[0] == parts[1]:
                    raise ValueError(f"Duplicate feature in pair: {item}")
                pairs.append((parts[0], parts[1]))
            return pairs
        return list(combinations(self.retained_features, 2))

    def _save_feature_availability(self) -> None:
        if self.clean_df is None:
            return
        rows = []
        for feat in self.retained_features:
            rows.append({
                "feature": feat,
                "non_null_count": int(self.clean_df[feat].notna().sum()),
                "non_null_ratio": float(self.clean_df[feat].notna().mean()),
                "mean": float(self.clean_df[feat].mean()),
                "std": float(self.clean_df[feat].std(ddof=1)),
            })
        pd.DataFrame(rows).sort_values(by="non_null_ratio", ascending=False).to_csv(
            self.out_dir / "feature_availability_summary.csv", index=False
        )

    # ------------------------------------------------------------------
    # Modeling
    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }

    def _pair_dataframe(self, pair: Tuple[str, str]) -> pd.DataFrame:
        if self.clean_df is None:
            raise RuntimeError("clean_data() must run first.")
        cols = [self.cfg.target_col, "canonical_smiles", *pair]
        if self.cfg.name_col in self.clean_df.columns:
            cols.append(self.cfg.name_col)
        return self.clean_df[cols].dropna(subset=[self.cfg.target_col, *pair]).reset_index(drop=True)

    def _evaluate_pair_model(self, pair: Tuple[str, str], model_name: str) -> Dict[str, object]:
        work = self._pair_dataframe(pair)
        X = work[list(pair)].values
        y = work[self.cfg.target_col].values.astype(float)

        rkf = RepeatedKFold(
            n_splits=self.cfg.cv_splits,
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
            "feature_1": pair[0],
            "feature_2": pair[1],
            "model": model_name,
            "n_samples_used": len(work),
            "cv_mae_mean": fold_df["mae"].mean(),
            "cv_mae_std": fold_df["mae"].std(ddof=1),
            "cv_rmse_mean": fold_df["rmse"].mean(),
            "cv_r2_mean": fold_df["r2"].mean(),
        }

    def screen(self) -> pd.DataFrame:
        rows = []
        total = len(self.feature_pairs) * len(self.cfg.models)
        idx = 0
        for pair in self.feature_pairs:
            for model_name in self.cfg.models:
                idx += 1
                print(f"[Screening] {idx:3d}/{total:3d} | model={model_name:<10} | pair={pair[0]} + {pair[1]}")
                rows.append(self._evaluate_pair_model(pair, model_name))

        self.ranking_df = pd.DataFrame(rows).sort_values(
            by=["cv_mae_mean", "cv_mae_std", "cv_rmse_mean"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        self.ranking_df.to_csv(self.out_dir / "feature_pair_ranking.csv", index=False)
        self._save_top_combinations_plot()
        return self.ranking_df

    def select_top_pairs(self) -> List[Tuple[str, str]]:
        if self.ranking_df is None:
            raise RuntimeError("screen() must run first.")
        best_per_pair = (
            self.ranking_df.sort_values(by=["cv_mae_mean", "cv_mae_std"])
            .groupby(["feature_1", "feature_2"], as_index=False)
            .first()
            .sort_values(by=["cv_mae_mean", "cv_mae_std"])
            .reset_index(drop=True)
        )
        top = best_per_pair.head(self.cfg.top_k_pairs)
        top.to_csv(self.out_dir / "top_unique_pairs.csv", index=False)
        return list(zip(top["feature_1"], top["feature_2"]))

    def choose_final_pair_and_model(self) -> Tuple[Tuple[str, str], str]:
        if self.ranking_df is None:
            raise RuntimeError("screen() must run first.")

        if self.cfg.holdout_pair is not None:
            parts = [x.strip() for x in self.cfg.holdout_pair.split(",")]
            if len(parts) != 2:
                raise ValueError("--holdout-pair must be feat1,feat2")
            pair = (parts[0], parts[1])
        else:
            best = self.ranking_df.iloc[0]
            pair = (best["feature_1"], best["feature_2"])

        if self.cfg.holdout_model is not None:
            model_name = self.cfg.holdout_model
        else:
            subset = self.ranking_df[
                (self.ranking_df["feature_1"] == pair[0]) &
                (self.ranking_df["feature_2"] == pair[1])
            ].sort_values(by=["cv_mae_mean", "cv_mae_std"])
            model_name = str(subset.iloc[0]["model"])

        return pair, model_name

    def fit_holdout(self, pair: Tuple[str, str], model_name: str) -> HoldoutResult:
        work = self._pair_dataframe(pair)
        y_bins = pd.qcut(work[self.cfg.target_col], q=min(5, work[self.cfg.target_col].nunique()), duplicates="drop")
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
        test_df["y_pred"] = pred_test
        test_df["abs_error"] = np.abs(test_df[self.cfg.target_col] - test_df["y_pred"])

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
    # Explanation
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
                background = shap.sample(
                    pd.DataFrame(X_train, columns=feature_names),
                    min(50, len(X_train)),
                )
                explain_df = pd.DataFrame(X_test, columns=feature_names)
                explainer = shap.KernelExplainer(
                    lambda z: model.predict(np.asarray(z)),
                    background,
                )
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

    def explain_top_pairs(self, top_pairs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
        if self.ranking_df is None:
            raise RuntimeError("screen() must run first.")

        rows = []
        root = self.out_dir / "top_pairs_explanations"
        root.mkdir(parents=True, exist_ok=True)

        for rank, pair in enumerate(list(top_pairs)[: self.cfg.top_pairs_to_explain], start=1):
            subset = self.ranking_df[
                (self.ranking_df["feature_1"] == pair[0]) &
                (self.ranking_df["feature_2"] == pair[1])
            ].sort_values(by=["cv_mae_mean", "cv_mae_std"])
            if subset.empty:
                continue

            model_name = str(subset.iloc[0]["model"])
            result = self.fit_holdout(pair, model_name)
            imp_df = self.explain(result)
            imp_df["pair"] = f"{pair[0]} + {pair[1]}"
            imp_df["model"] = model_name
            rows.append(imp_df)

            pair_dir = root / f"{rank:02d}__{pair[0]}__{pair[1]}__{model_name}".replace("/", "_")
            pair_dir.mkdir(parents=True, exist_ok=True)
            imp_df.to_csv(pair_dir / "explanation.csv", index=False)
            self._save_explanation_bar(imp_df, pair_dir / "explanation.png", f"{pair[0]} + {pair[1]} | {model_name}")

            shap_payload, shap_plot_type = self._compute_shap_payload(result)

            if shap_payload is not None:
                if self.cfg.save_shap_beeswarm:
                    self._save_shap_beeswarm_plot(
                        shap_payload,
                        pair_dir / "shap_beeswarm.png",
                        f"{pair[0]} + {pair[1]} | {model_name} | {shap_plot_type}",
                    )
                    self._save_shap_bar_plot(
                        shap_payload,
                        pair_dir / "shap_bar.png",
                        f"{pair[0]} + {pair[1]} | {model_name} | {shap_plot_type}",
                    )

                if self.cfg.save_shap_dependence:
                    self._save_shap_dependence_plots(
                        shap_payload,
                        pair_dir / "shap_dependence",
                        f"{pair[0]} + {pair[1]} | {model_name}",
                    )

        if not rows:
            return pd.DataFrame()

        all_imp = pd.concat(rows, ignore_index=True)
        all_imp.to_csv(self.out_dir / "top_pairs_explanation_long.csv", index=False)
        return all_imp

    # ------------------------------------------------------------------
    # Saving / plotting
    # ------------------------------------------------------------------
    def _save_top_combinations_plot(self) -> None:
        if self.ranking_df is None or self.ranking_df.empty:
            return
        top = self.ranking_df.head(min(12, len(self.ranking_df))).copy()
        labels = top.apply(lambda r: f"{r['model']} | {r['feature_1']} + {r['feature_2']}", axis=1).tolist()

        plt.figure(figsize=(12, max(4.5, 0.45 * len(top))))
        plt.barh(range(len(top))[::-1], top["cv_mae_mean"].values, xerr=top["cv_mae_std"].values)
        plt.yticks(range(len(top))[::-1], labels)
        plt.xlabel("Repeated CV MAE")
        plt.title("Top model + feature-pair combinations")
        plt.grid(True, axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out_dir / "top_model_pair_combinations.png", dpi=150)
        plt.close()

    def _save_holdout_parity_plot(self, result: HoldoutResult) -> None:
        y_tr = result.train_df[result.target_col].values
        yp_tr = result.train_df["y_pred"].values
        y_te = result.test_df[result.target_col].values
        yp_te = result.test_df["y_pred"].values

        all_vals = np.concatenate([y_tr, yp_tr, y_te, yp_te])
        lo, hi = float(np.min(all_vals) - 5), float(np.max(all_vals) + 5)

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
        plt.savefig(self.out_dir / "best_model_holdout_parity.png", dpi=150)
        plt.close()

    def _save_explanation_bar(self, df: pd.DataFrame, out_png: Path, title: str) -> None:
        plot_df = df.sort_values(by="importance", ascending=True)
        plt.figure(figsize=(7, max(3.5, 0.6 * len(plot_df))))
        plt.barh(plot_df["feature"], plot_df["importance"].values, xerr=plot_df["importance_std"].values)
        plt.xlabel("Mean absolute contribution / importance")
        plt.title(title)
        plt.grid(True, axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    def save_holdout_outputs(self, result: HoldoutResult) -> None:
        result.train_df.to_csv(self.out_dir / "holdout_predictions_train_best_model.csv", index=False)
        result.test_df.to_csv(self.out_dir / "holdout_predictions_best_model.csv", index=False)

        cols = [
            c for c in [
                self.cfg.name_col,
                "canonical_smiles",
                *result.feature_pair,
                result.target_col,
                "y_pred",
                "abs_error",
            ] if c in result.test_df.columns
        ]
        worst = result.test_df.sort_values(by="abs_error", ascending=False).head(15)[cols]
        worst.to_csv(self.out_dir / "worst_predictions_best_model.csv", index=False)

        pd.DataFrame([
            {
                "model": result.model_name,
                "feature_1": result.feature_pair[0],
                "feature_2": result.feature_pair[1],
                "mae_train": result.mae_train,
                "mae_test": result.mae_test,
                "rmse_test": result.rmse_test,
                "r2_test": result.r2_test,
                "n_train": len(result.train_df),
                "n_test": len(result.test_df),
            }
        ]).to_csv(self.out_dir / "best_model_holdout_summary.csv", index=False)

        self._save_holdout_parity_plot(result)

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
            json.dump(config_dict, f, indent=2)

        if HAS_YAML:
            with open(self.out_dir / "run_config_used.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    def run(self) -> HoldoutResult:
        print("=" * 80)
        print("Cp experiment framework (class version, YAML-enabled)")
        print("=" * 80)
        print(f"Config file    : {self.cfg.config_file}")
        print(f"Data file      : {self.data_path}")
        print(f"Output dir     : {self.out_dir}")
        print(f"Target column  : {self.cfg.target_col}")
        print(f"Models         : {list(self.cfg.models)}")
        print(f"Random seed    : {self.cfg.seed}")
        print()

        self.load_data()
        self.clean_data()
        assert self.raw_df is not None
        assert self.clean_df is not None

        print(f"[1] Raw shape: {self.raw_df.shape}")
        print(f"[1] Clean shape: {self.clean_df.shape}")
        print(f"[1] Retained features: {self.retained_features}")
        print(f"[2] Number of feature pairs: {len(self.feature_pairs)}")

        ranking_df = self.screen()
        print()
        print("[3] Top 10 screening results:")
        print(ranking_df.head(10).to_string(index=False))

        top_pairs = self.select_top_pairs()
        print()
        print(f"[4] Top unique feature pairs: {top_pairs}")

        if self.cfg.explain_top_pairs:
            self.explain_top_pairs(top_pairs)

        final_pair, final_model = self.choose_final_pair_and_model()
        print(f"[5] Final holdout choice: model={final_model}, pair={final_pair[0]} + {final_pair[1]}")

        result = self.fit_holdout(final_pair, final_model)
        self.save_holdout_outputs(result)
        self.save_run_config()

        print()
        print("=" * 80)
        print("Finished.")
        print("=" * 80)
        print(f"Best holdout model  : {result.model_name}")
        print(f"Best holdout pair   : {result.feature_pair[0]} + {result.feature_pair[1]}")
        print(f"Holdout test MAE    : {result.mae_test:.4f}")
        print(f"Holdout test RMSE   : {result.rmse_test:.4f}")
        print(f"Holdout test R2     : {result.r2_test:.4f}")
        print(f"All outputs saved to: {self.out_dir}")
        return result


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for --config support. Install it with: pip install pyyaml"
        )

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


def parse_args() -> ExperimentConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Lean class-based Cp experiment framework (YAML-first)",
        parents=[pre_parser],
    )
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--smiles-col", type=str, default=None)
    parser.add_argument("--name-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--candidate-features", nargs="*", default=None)
    parser.add_argument("--fixed-pairs", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--top-k-pairs", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-non-null-ratio", type=float, default=None)
    parser.add_argument("--iqr-outlier-threshold", type=float, default=None)
    parser.add_argument("--top-pairs-to-explain", type=int, default=None)
    parser.add_argument("--holdout-model", type=str, default=None, choices=DEFAULT_MODELS)
    parser.add_argument("--holdout-pair", type=str, default=None)

    parser.add_argument("--explain-top-pairs", dest="explain_top_pairs", action="store_true")
    parser.add_argument("--no-explain-top-pairs", dest="explain_top_pairs", action="store_false")
    parser.set_defaults(explain_top_pairs=None)

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

    config = ExperimentConfig(config_file=pre_args.config)

    yaml_data: Dict[str, Any] = {}
    if pre_args.config is not None:
        yaml_data = load_yaml_config(pre_args.config)

    valid_fields = {f.name for f in fields(ExperimentConfig)}
    unknown_yaml_keys = [k for k in yaml_data if k not in valid_fields]
    if unknown_yaml_keys:
        raise ValueError(f"Unknown YAML keys: {unknown_yaml_keys}")

    for f in fields(ExperimentConfig):
        if f.name in yaml_data:
            setattr(config, f.name, yaml_data[f.name])

    for f in fields(ExperimentConfig):
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
    experiment = CpExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()

