from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .data_prep import make_task_dataset, read_raw_excel, save_dataset_summary
from .descriptors import featurize_dataframe
from .evaluate import cross_validate_regressor
from .models import build_regressor
from .plotting import save_parity_plot, save_summary_barplot


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return only columns that exist in df."""
    return [col for col in columns if col in df.columns]


def run_feature_search(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run feature/model search for Cp prediction.

    This feature-search pipeline is phase-free:
        raw Excel -> SMILES-derived descriptors -> Cp regression benchmark

    Outputs:
        output/feature_search/dataset_summary.csv
        output/feature_search/feature_search_summary.csv
        output/feature_search/feature_search_summary_ranked.csv
        output/feature_search/oof_predictions/*.csv
        output/feature_search/parity_plots/*.png
        output/feature_search/top20_feature_search_mae.png
        models/feature_search/best_feature_search_model.joblib
    """
    output_dir = Path(config.get("output_dir", "output/feature_search"))
    model_dir = Path(config.get("model_dir", "models/feature_search"))

    oof_dir = output_dir / "oof_predictions"
    parity_dir = output_dir / "parity_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)
    parity_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_raw_excel(config)
    save_dataset_summary(raw_df, output_dir)

    tasks = config.get("tasks", ["all"])
    feature_sets = config.get("feature_sets", ["basic_2"])
    model_names = config.get("models", ["ridge"])

    # Rule-compliant mode: do not allow phase-specific feature search.
    invalid_tasks = [task for task in tasks if task != "all"]
    if invalid_tasks:
        raise ValueError(
            "Phase-specific feature search is disabled for the rule-compliant pipeline. "
            f"Use only tasks: ['all']. Invalid tasks: {invalid_tasks}"
        )

    summary_rows = []
    best = None

    for task in tasks:
        task_df = make_task_dataset(raw_df, task)

        if len(task_df) < 10:
            print(f"[WARN] Skip task={task}: too few rows n={len(task_df)}")
            continue

        for feature_set in feature_sets:
            try:
                feat_df, feature_names = featurize_dataframe(task_df, feature_set, config)
            except Exception as exc:
                print(f"[WARN] Skip task={task}, feature_set={feature_set}: {exc}")
                continue

            if len(feat_df) < 10:
                print(f"[WARN] Skip task={task}, feature_set={feature_set}: valid rows n={len(feat_df)}")
                continue

            X = feat_df[feature_names]
            y = feat_df["Cp_J_molK"].astype(float)
            groups = feat_df["canonical_smiles"].astype(str) if "canonical_smiles" in feat_df.columns else None

            for model_name in model_names:
                model = build_regressor(model_name, config)

                try:
                    metrics, y_oof = cross_validate_regressor(
                        model=model,
                        X=X,
                        y=y,
                        groups=groups,
                        config=config,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] Failed task={task}, "
                        f"feature_set={feature_set}, model={model_name}: {exc}"
                    )
                    continue

                row = {
                    "task": task,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_rows": int(len(feat_df)),
                    "n_features": int(len(feature_names)),
                    "mae_cv": metrics["mae"],
                    "rmse_cv": metrics["rmse"],
                    "r2_cv": metrics["r2"],
                    "cv_type": metrics["cv_type"],
                    "n_splits": metrics["n_splits"],
                }
                summary_rows.append(row)

                keep_cols = _existing_columns(
                    feat_df,
                    [
                        "task",
                        "phase",          # compatibility only; normally "all"
                        "compound_name",
                        "canonical_smiles",
                        "smiles_raw",
                        "Cp_J_molK",
                        "source",
                        "source_sheet",
                    ],
                )

                oof_df = feat_df[keep_cols].copy()
                oof_df["feature_set"] = feature_set
                oof_df["model"] = model_name
                oof_df["task"] = task
                oof_df["Cp_pred_oof"] = y_oof
                oof_df["abs_error"] = (oof_df["Cp_J_molK"] - oof_df["Cp_pred_oof"]).abs()

                safe_name = f"{task}__{feature_set}__{model_name}"
                oof_path = oof_dir / f"{safe_name}_oof.csv"
                oof_df.to_csv(oof_path, index=False)

                save_parity_plot(
                    oof_df,
                    output_path=parity_dir / f"{safe_name}_parity.png",
                    title=f"{task} | {feature_set} | {model_name}",
                )

                if best is None or row["mae_cv"] < best["row"]["mae_cv"]:
                    best = {
                        "row": row,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "task": task,
                        "feature_names": feature_names,
                        "feat_df": feat_df,
                    }

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "feature_search_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not summary_df.empty:
        ranked = summary_df.sort_values("mae_cv").reset_index(drop=True)
        ranked.to_csv(output_dir / "feature_search_summary_ranked.csv", index=False)
        save_summary_barplot(ranked, output_dir)

    if best is not None:
        feature_names = best["feature_names"]
        feat_df = best["feat_df"]

        X = feat_df[feature_names]
        y = feat_df["Cp_J_molK"].astype(float)

        final_model = build_regressor(best["model_name"], config)
        final_model.fit(X, y)

        best_model_path = model_dir / "best_feature_search_model.joblib"
        joblib.dump(final_model, best_model_path)

        best_info = {
            **best["row"],
            "feature_names": feature_names,
            "model_path": str(best_model_path),
        }
        pd.DataFrame([best_info]).to_csv(output_dir / "best_feature_search_model.csv", index=False)

    return {
        "summary_path": str(summary_path),
        "n_experiments": int(len(summary_rows)),
        "best": best["row"] if best is not None else None,
    }
