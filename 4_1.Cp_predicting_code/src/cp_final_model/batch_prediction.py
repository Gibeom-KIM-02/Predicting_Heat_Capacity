from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .compound_resolver import resolve_name_to_smiles
from .features import canonicalize_smiles, featurize_smiles_list


# =============================================================================
# Basic helpers
# =============================================================================

def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first existing column from candidates."""
    col_map = {str(c).strip(): c for c in df.columns}

    for cand in candidates:
        if cand in col_map:
            return col_map[cand]

    return None


def _normalize_name(name: Any) -> str:
    """
    Normalize compound names for robust lookup.

    This is intentionally conservative:
    - strip leading/trailing spaces
    - lowercase
    - collapse repeated spaces
    - normalize common unicode dashes
    - normalize a few Greek letters
    """
    if pd.isna(name):
        return ""

    text = str(name).strip().lower()
    text = re.sub(r"\s+", " ", text)

    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("α", "alpha")
    text = text.replace("β", "beta")
    text = text.replace("γ", "gamma")

    return text


def _rmse(y_true, y_pred) -> float:
    """Return RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _read_table(path: Path, sheet_name=None) -> pd.DataFrame:
    """Read Excel or CSV table."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        if sheet_name is None:
            return pd.read_excel(path, sheet_name=0)
        return pd.read_excel(path, sheet_name=sheet_name)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format: {path.suffix}")


# =============================================================================
# Model loading
# =============================================================================

def _load_model_bundle(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load trained final Cp regressor and feature schema."""
    model_dir = Path(config["model_dir"])

    model_path = model_dir / "cp_regressor.joblib"
    schema_path = model_dir / "feature_schema.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    if not schema_path.exists():
        raise FileNotFoundError(f"Missing feature schema: {schema_path}")

    model = joblib.load(model_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return {
        "model": model,
        "schema": schema,
    }


# =============================================================================
# Name -> SMILES lookup builders
# =============================================================================

def _build_local_name_to_smiles_lookup(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build local name -> canonical SMILES lookup from processed/training files.

    This uses only name-SMILES mapping. It does not use Cp values as features.
    Priority use case:
        external name-only file contains compounds already present in the
        processed dataset.
    """
    resolver_cfg = config.get("name_resolution", {})

    if not bool(resolver_cfg.get("use_local_lookup", False)):
        return {}

    lookup_files = resolver_cfg.get("local_lookup_files", []) or []
    sheet_name = resolver_cfg.get("sheet_name", None)

    name_candidates = config.get("name_columns", [])
    smiles_candidates = config.get("smiles_columns", [])

    lookup: Dict[str, str] = {}

    for file_path in lookup_files:
        path = Path(file_path)

        if not path.exists():
            print(f"[WARN] Local lookup file not found: {path}")
            continue

        try:
            df = _read_table(path, sheet_name=sheet_name)
        except Exception as exc:
            print(f"[WARN] Failed to read local lookup file {path}: {exc}")
            continue

        df.columns = [str(c).strip() for c in df.columns]

        name_col = _first_existing_column(df, name_candidates)
        smiles_col = _first_existing_column(df, smiles_candidates)

        if name_col is None or smiles_col is None:
            print(
                f"[WARN] Local lookup file lacks name or SMILES columns: {path}\n"
                f"       name_col={name_col}, smiles_col={smiles_col}\n"
                f"       columns={list(df.columns)}"
            )
            continue

        for _, row in df.iterrows():
            name_key = _normalize_name(row.get(name_col))
            smiles = row.get(smiles_col)

            if not name_key:
                continue
            if pd.isna(smiles) or not str(smiles).strip():
                continue

            canonical, valid = canonicalize_smiles(str(smiles).strip())
            if not valid or canonical is None:
                continue

            # Keep first occurrence for stability.
            if name_key not in lookup:
                lookup[name_key] = canonical

    print(f"[INFO] Loaded local name→SMILES lookup entries: {len(lookup)}")
    return lookup


def _build_manual_name_to_smiles_map(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build manual name -> canonical SMILES lookup from a small correction CSV.

    Expected columns:
        compound_name, smiles

    This is useful for a few unresolved or ambiguous names.
    """
    resolver_cfg = config.get("name_resolution", {})
    map_file = resolver_cfg.get("manual_map_file")

    if not map_file:
        return {}

    path = Path(map_file)

    if not path.exists():
        print(f"[WARN] Manual name-SMILES map file not found: {path}")
        return {}

    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip() for c in df.columns]

    if "compound_name" not in df.columns or "smiles" not in df.columns:
        raise ValueError("Manual map must contain columns: compound_name, smiles")

    lookup: Dict[str, str] = {}

    for _, row in df.iterrows():
        name_key = _normalize_name(row.get("compound_name"))
        smiles = row.get("smiles")

        if not name_key:
            continue
        if pd.isna(smiles) or not str(smiles).strip():
            continue

        canonical, valid = canonicalize_smiles(str(smiles).strip())
        if valid and canonical is not None:
            lookup[name_key] = canonical

    print(f"[INFO] Loaded manual name→SMILES entries: {len(lookup)}")
    return lookup


def _resolve_name_with_priority(
    name: Any,
    local_lookup: Dict[str, str],
    manual_lookup: Dict[str, str],
    config: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    """
    Resolve compound name to canonical SMILES using priority:

    1. local processed dataset lookup
    2. manual correction map
    3. PubChem/PUG-REST resolver
    """
    name_key = _normalize_name(name)

    if not name_key:
        return None, "empty_name"

    if name_key in local_lookup:
        return local_lookup[name_key], "resolved_from_local_lookup"

    if name_key in manual_lookup:
        return manual_lookup[name_key], "resolved_from_manual_map"

    resolver_cfg = config.get("name_resolution", {})
    allow_pubchem = bool(resolver_cfg.get("allow_pubchem_fallback", True))

    if allow_pubchem:
        resolved = resolve_name_to_smiles(str(name).strip())
        if resolved:
            canonical, valid = canonicalize_smiles(resolved)
            if valid and canonical is not None:
                return canonical, "resolved_from_pubchem"

    return None, "name_resolution_failed"


# =============================================================================
# Plotting
# =============================================================================

def _save_parity_plot(
    pred_df: pd.DataFrame,
    output_path: Path,
    true_col: str = "Cp_true_J_molK",
    pred_col: str = "Cp_pred_J_molK",
) -> None:
    """Save parity plot if true Cp values exist."""
    if true_col not in pred_df.columns or pred_col not in pred_df.columns:
        return

    plot_df = pred_df[[true_col, pred_col]].dropna().copy()

    if plot_df.empty:
        return

    y_true = plot_df[true_col].astype(float).values
    y_pred = plot_df[pred_col].astype(float).values

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    mae = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(y_true, y_pred, s=32, alpha=0.80)
    plt.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("True Cp [J/mol*K]")
    plt.ylabel("Predicted Cp [J/mol*K]")
    plt.title("External Test Set Cp Prediction")

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
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# Main external prediction
# =============================================================================

def predict_external_test_file(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict Cp for an external test Excel/CSV file.

    Required:
        - SMILES column OR compound name column

    Optional:
        - true Cp column
          If present, MAE/RMSE/R2 are computed.

    Name-only mode:
        If SMILES is absent, name is resolved using:
            local processed lookup -> manual map -> PubChem/PUG-REST
    """
    input_path = Path(config["external_test"]["input_file"])
    output_dir = Path(config["external_test"].get("output_dir", "output/external_test"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"External test file not found: {input_path}")

    # -------------------------------------------------------------------------
    # Read external file
    # -------------------------------------------------------------------------
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        sheet_name = config["external_test"].get("sheet_name", 0)
        raw = pd.read_excel(input_path, sheet_name=sheet_name)
    elif input_path.suffix.lower() == ".csv":
        raw = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported external test file format: {input_path.suffix}")

    raw.columns = [str(c).strip() for c in raw.columns]

    smiles_col = _first_existing_column(raw, config.get("smiles_columns", []))
    name_col = _first_existing_column(raw, config.get("name_columns", []))
    target_col = _first_existing_column(raw, config.get("target_columns", []))

    if smiles_col is None and name_col is None:
        raise ValueError(
            "External test file must contain either a SMILES column or a compound name column. "
            f"Actual columns: {list(raw.columns)}"
        )

    # -------------------------------------------------------------------------
    # Build name resolution lookups
    # -------------------------------------------------------------------------
    local_lookup = _build_local_name_to_smiles_lookup(config)
    manual_lookup = _build_manual_name_to_smiles_map(config)

    # -------------------------------------------------------------------------
    # Load trained model
    # -------------------------------------------------------------------------
    bundle = _load_model_bundle(config)
    model = bundle["model"]
    schema = bundle["schema"]

    feature_set = schema["feature_set"]
    feature_names = schema["feature_names"]
    required_extra = schema.get("extra_feature_columns", []) or []

    rows = []

    # -------------------------------------------------------------------------
    # Predict row by row
    # -------------------------------------------------------------------------
    for idx, row in raw.iterrows():
        input_name = row.get(name_col) if name_col else np.nan
        input_smiles = row.get(smiles_col) if smiles_col else np.nan

        resolved_smiles = None
        resolve_status = "not_attempted"

        # 1. Prefer explicit SMILES column.
        if pd.notna(input_smiles) and str(input_smiles).strip():
            resolved_smiles = str(input_smiles).strip()
            resolve_status = "from_smiles_column"

        # 2. If no SMILES, resolve from name.
        elif pd.notna(input_name) and str(input_name).strip():
            resolved_smiles, resolve_status = _resolve_name_with_priority(
                input_name,
                local_lookup=local_lookup,
                manual_lookup=manual_lookup,
                config=config,
            )

        cp_true_raw = row.get(target_col) if target_col else np.nan
        cp_true = pd.to_numeric(pd.Series([cp_true_raw]), errors="coerce").iloc[0]

        if resolved_smiles is None:
            rows.append(
                {
                    "row_index": idx,
                    "compound_name": input_name,
                    "input_smiles": input_smiles,
                    "resolved_smiles": np.nan,
                    "canonical_smiles": np.nan,
                    "resolve_status": resolve_status,
                    "Cp_true_J_molK": cp_true,
                    "Cp_pred_J_molK": np.nan,
                    "abs_error": np.nan,
                    "error": "SMILES unavailable",
                }
            )
            continue

        canonical, valid = canonicalize_smiles(resolved_smiles)

        if not valid or canonical is None:
            rows.append(
                {
                    "row_index": idx,
                    "compound_name": input_name,
                    "input_smiles": input_smiles,
                    "resolved_smiles": resolved_smiles,
                    "canonical_smiles": np.nan,
                    "resolve_status": resolve_status,
                    "Cp_true_J_molK": cp_true,
                    "Cp_pred_J_molK": np.nan,
                    "abs_error": np.nan,
                    "error": "Invalid SMILES after resolution",
                }
            )
            continue

        X = featurize_smiles_list([canonical], feature_set=feature_set)

        for col in required_extra:
            if col not in raw.columns:
                raise ValueError(
                    f"Model requires extra feature '{col}', but external test file does not contain it."
                )

            X[col] = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]

        X = X[feature_names]

        cp_pred = float(model.predict(X)[0])

        rows.append(
            {
                "row_index": idx,
                "compound_name": input_name,
                "input_smiles": input_smiles,
                "resolved_smiles": resolved_smiles,
                "canonical_smiles": canonical,
                "resolve_status": resolve_status,
                "Cp_true_J_molK": cp_true,
                "Cp_pred_J_molK": cp_pred,
                "abs_error": abs(cp_true - cp_pred) if pd.notna(cp_true) else np.nan,
                "error": "",
            }
        )

    pred_df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------
    pred_path = output_dir / "external_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "input_file": str(input_path),
        "n_rows": int(len(pred_df)),
        "n_predicted": int(pred_df["Cp_pred_J_molK"].notna().sum()),
        "n_failed": int(pred_df["Cp_pred_J_molK"].isna().sum()),
        "model_dir": str(config["model_dir"]),
        "feature_set": feature_set,
        "smiles_col_used": smiles_col if smiles_col else "",
        "name_col_used": name_col if name_col else "",
        "target_col_used": target_col if target_col else "",
    }

    if "resolve_status" in pred_df.columns:
        status_counts = pred_df["resolve_status"].value_counts(dropna=False).to_dict()
        for status, count in status_counts.items():
            summary[f"resolve_status__{status}"] = int(count)

    eval_df = pred_df[["Cp_true_J_molK", "Cp_pred_J_molK"]].dropna().copy()

    if not eval_df.empty:
        y_true = eval_df["Cp_true_J_molK"].astype(float).values
        y_pred = eval_df["Cp_pred_J_molK"].astype(float).values

        summary.update(
            {
                "n_evaluated": int(len(eval_df)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": _rmse(y_true, y_pred),
                "r2": float(r2_score(y_true, y_pred)) if len(eval_df) >= 2 else np.nan,
            }
        )

        _save_parity_plot(
            pred_df,
            output_path=output_dir / "parity_external_test.png",
        )
    else:
        summary.update(
            {
                "n_evaluated": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
            }
        )

    summary_path = output_dir / "external_test_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    failed_df = pred_df[pred_df["Cp_pred_J_molK"].isna()].copy()
    if not failed_df.empty:
        failed_df.to_csv(output_dir / "failed_external_predictions.csv", index=False)

    return {
        "prediction_path": str(pred_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }