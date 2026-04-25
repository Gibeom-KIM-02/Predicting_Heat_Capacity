#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cp_common.config import load_config
from cp_final_model.regressor import train_final_regressor


def _safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace(" ", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(":", "_")
    )


def _make_trials(base_config: dict) -> list[dict]:
    """
    Define final-model trial configurations.

    All trials use the same input data, split seed, and output structure.
    Only feature_set / regressor / sample_weighting are changed.
    """
    trials = []

    def add_trial(
        name: str,
        feature_set: str,
        regressor: dict,
        sample_weighting_enabled: bool = False,
    ) -> None:
        cfg = copy.deepcopy(base_config)
        cfg["feature_set"] = feature_set
        cfg["regressor"] = regressor

        cfg["sample_weighting"] = copy.deepcopy(
            base_config.get(
                "sample_weighting",
                {
                    "enabled": False,
                    "method": "cp_bin_inverse_frequency",
                    "n_bins": 10,
                },
            )
        )
        cfg["sample_weighting"]["enabled"] = bool(sample_weighting_enabled)

        trials.append(
            {
                "name": name,
                "config": cfg,
            }
        )

    # ------------------------------------------------------------------
    # RandomForest trials
    # ------------------------------------------------------------------
    add_trial(
        name="rf_selected_leaf1",
        feature_set="selected_rdkit_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    add_trial(
        name="rf_selected_leaf2",
        feature_set="selected_rdkit_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    add_trial(
        name="rf_selected_depth20_leaf1",
        feature_set="selected_rdkit_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": 20,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    add_trial(
        name="rf_small_leaf1",
        feature_set="rdkit_small",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    add_trial(
        name="rf_small_leaf2",
        feature_set="rdkit_small",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    add_trial(
        name="rf_shap_top10_leaf1",
        feature_set="shap_top10_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )
    
    add_trial(
        name="rf_shap_top15_leaf1",
        feature_set="shap_top15_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )
    
    add_trial(
        name="rf_chem_compact_leaf1",
        feature_set="chem_compact_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    # ------------------------------------------------------------------
    # GradientBoosting trials
    # ------------------------------------------------------------------
    add_trial(
        name="gb_selected_lr003",
        feature_set="selected_rdkit_v1",
        regressor={
            "model_type": "gradient_boosting",
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "subsample": 0.9,
        },
    )

    add_trial(
        name="gb_small_lr003",
        feature_set="rdkit_small",
        regressor={
            "model_type": "gradient_boosting",
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "subsample": 0.9,
        },
    )

    # ------------------------------------------------------------------
    # Optional sample-weighting trials
    # ------------------------------------------------------------------
    add_trial(
        name="rf_selected_leaf2_weighted",
        feature_set="selected_rdkit_v1",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        sample_weighting_enabled=True,
    )

    add_trial(
        name="rf_small_leaf2_weighted",
        feature_set="rdkit_small",
        regressor={
            "model_type": "random_forest",
            "n_estimators": 800,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "max_depth": None,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        sample_weighting_enabled=True,
    )

    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_train.yaml")
    parser.add_argument("--out-dir", default="output/final_model_sweep")
    parser.add_argument(
        "--copy-best-to-final",
        action="store_true",
        help="Copy best trial model/schema to models/final_model after sweep.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)

    sweep_dir = Path(args.out_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    trials = _make_trials(base_config)

    summary_rows = []

    print("=" * 80)
    print("Final model sweep")
    print("=" * 80)
    print(f"Number of trials: {len(trials)}")
    print(f"Sweep output dir : {sweep_dir}")

    for i, trial in enumerate(trials, start=1):
        trial_name = trial["name"]
        safe_trial_name = _safe_name(f"trial_{i:03d}_{trial_name}")

        trial_out_dir = sweep_dir / safe_trial_name / "output"
        trial_model_dir = sweep_dir / safe_trial_name / "models"

        cfg = copy.deepcopy(trial["config"])
        cfg["output_dir"] = str(trial_out_dir)
        cfg["model_dir"] = str(trial_model_dir)

        # Training logs inside each trial folder.
        if "training_log" in cfg:
            cfg["training_log"] = copy.deepcopy(cfg["training_log"])
            cfg["training_log"]["output_dir"] = str(trial_out_dir / "training_logs")

        print()
        print("-" * 80)
        print(f"[{i}/{len(trials)}] Running {trial_name}")
        print("-" * 80)
        print("feature_set:", cfg["feature_set"])
        print("regressor  :", json.dumps(cfg["regressor"], indent=2))
        print("weighting  :", cfg.get("sample_weighting", {}).get("enabled", False))

        try:
            result = train_final_regressor(cfg)
            status = "success"
            error = ""
        except Exception as exc:
            result = {}
            status = "failed"
            error = repr(exc)
            print(f"[ERROR] Trial failed: {error}")

        row = {
            "trial_index": i,
            "trial_name": trial_name,
            "status": status,
            "error": error,
            "feature_set": cfg.get("feature_set"),
            "model_type": cfg.get("regressor", {}).get("model_type"),
            "sample_weighting_enabled": cfg.get("sample_weighting", {}).get("enabled", False),
            "trial_output_dir": str(trial_out_dir),
            "trial_model_dir": str(trial_model_dir),
            **result,
        }

        # Add important regressor hyperparameters as flat columns.
        for k, v in cfg.get("regressor", {}).items():
            row[f"regressor__{k}"] = v

        summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(sweep_dir / "sweep_summary_partial.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(sweep_dir / "sweep_summary.csv", index=False)

    success_df = summary_df[summary_df["status"].eq("success")].copy()

    if success_df.empty:
        print("\nNo successful trials.")
        return

    # Ranking priority:
    # 1. test_mae, if available
    # 2. validation_mae
    # 3. mae_oof
    rank_col = None
    for candidate in ["test_mae", "validation_mae", "mae_oof"]:
        if candidate in success_df.columns and success_df[candidate].notna().any():
            rank_col = candidate
            break

    if rank_col is None:
        print("\nNo ranking metric found.")
        return

    ranked = success_df.sort_values(rank_col).reset_index(drop=True)
    ranked.to_csv(sweep_dir / "sweep_summary_ranked.csv", index=False)

    best = ranked.iloc[0].to_dict()

    print()
    print("=" * 80)
    print("Sweep finished.")
    print("=" * 80)
    print(f"Ranking metric: {rank_col}")
    print("\nTop trials:")
    display_cols = [
        "trial_index",
        "trial_name",
        "feature_set",
        "model_type",
        "sample_weighting_enabled",
        "mae_oof",
        "validation_mae",
        "test_mae",
        "train_mae",
        "test_r2",
    ]
    display_cols = [c for c in display_cols if c in ranked.columns]
    print(ranked[display_cols].head(20).to_string(index=False))

    print()
    print("Best trial:")
    for key in [
        "trial_index",
        "trial_name",
        "feature_set",
        "model_type",
        "mae_oof",
        "validation_mae",
        "test_mae",
        "test_r2",
        "trial_model_dir",
    ]:
        if key in best:
            print(f"  {key}: {best[key]}")

    if args.copy_best_to_final:
        final_model_dir = Path(base_config.get("model_dir", "models/final_model"))
        final_model_dir.mkdir(parents=True, exist_ok=True)

        best_model_dir = Path(best["trial_model_dir"])

        for filename in ["cp_regressor.joblib", "feature_schema.json"]:
            src = best_model_dir / filename
            dst = final_model_dir / filename

            if src.exists():
                shutil.copy2(src, dst)
                print(f"[COPY] {src} -> {dst}")
            else:
                print(f"[WARN] Missing best artifact: {src}")

        print(f"\nBest model copied to: {final_model_dir}")


if __name__ == "__main__":
    main()
