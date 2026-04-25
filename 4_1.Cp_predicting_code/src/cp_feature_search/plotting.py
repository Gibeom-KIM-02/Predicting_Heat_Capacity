from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_summary_barplot(summary_df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Save a simple MAE ranking plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        return

    plot_df = summary_df.sort_values("mae_cv").head(20).copy()
    labels = plot_df["task"] + " | " + plot_df["feature_set"] + " | " + plot_df["model"]

    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
    plt.barh(labels, plot_df["mae_cv"])
    plt.xlabel("CV MAE [J/mol*K]")
    plt.ylabel("Experiment")
    plt.title("Top Feature Search Results")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "top20_feature_search_mae.png", dpi=300)
    plt.close()


def save_parity_plot(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    """
    Save parity plot for one selected experiment.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = df[["Cp_J_molK", "Cp_pred_oof"]].dropna().copy()
    if tmp.empty:
        return

    y_true = tmp["Cp_J_molK"].astype(float).values
    y_pred = tmp["Cp_pred_oof"].astype(float).values

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    plt.xlabel("True Cp [J/mol*K]")
    plt.ylabel("OOF Predicted Cp [J/mol*K]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
