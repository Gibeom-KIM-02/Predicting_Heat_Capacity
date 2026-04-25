from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_final_report_plots(config: Dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_path = output_dir / "final_regressor_oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Missing {oof_path}. Run scripts/02_train_final_regressor.py first."
        )

    df = pd.read_csv(oof_path)
    tmp = df[["Cp_J_molK", "Cp_pred_oof"]].dropna().copy()

    y_true = tmp["Cp_J_molK"].astype(float).values
    y_pred = tmp["Cp_pred_oof"].astype(float).values

    summary = {
        "n": int(len(tmp)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(tmp) >= 2 else np.nan,
    }

    pd.DataFrame([summary]).to_csv(output_dir / "final_report_summary.csv", index=False)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)

    plt.xlabel("True Cp [J/mol*K]")
    plt.ylabel("OOF Predicted Cp [J/mol*K]")
    plt.title("Final Cp Regression Model")

    metric_text = (
        f"MAE  = {summary['mae']:.3f} J/mol*K\n"
        f"RMSE = {summary['rmse']:.3f} J/mol*K\n"
        f"R²   = {summary['r2']:.4f}\n"
        f"N    = {summary['n']}"
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
    plt.savefig(output_dir / "parity_final_regressor.png", dpi=300)
    plt.close()

