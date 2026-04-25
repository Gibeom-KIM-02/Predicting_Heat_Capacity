from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .features import canonicalize_smiles, featurize_smiles_list


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


def _read_one_sheet(input_excel: Path, sheet_name: str | int | None) -> pd.DataFrame:
    if sheet_name is None:
        return pd.read_excel(input_excel, sheet_name=0)
    return pd.read_excel(input_excel, sheet_name=sheet_name)


def read_raw_excel(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Read raw Cp dataset without requiring phase labels.

    Required information:
        compound name or SMILES
        Cp target

    Optional:
        source
        up to 2 extra physical features already present in Excel
    """
    input_excel = Path(config["input_excel"])
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_excel}")

    read_all_sheets = bool(config.get("read_all_sheets", False))
    sheet_name = config.get("sheet_name", None)

    if read_all_sheets:
        sheets = pd.read_excel(input_excel, sheet_name=None)
        frames = []
        for sname, df in sheets.items():
            df = df.copy()
            df["source_sheet"] = str(sname)
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = _read_one_sheet(input_excel, sheet_name)
        raw["source_sheet"] = str(sheet_name) if sheet_name is not None else "sheet0"

    raw.columns = [str(c).strip() for c in raw.columns]

    smiles_col = first_existing_column(raw, config["smiles_columns"])
    name_col = first_existing_column(raw, config["name_columns"])
    target_col = first_existing_column(raw, config["target_columns"])
    source_col = first_existing_column(raw, config.get("source_columns", []))

    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(raw.columns)}")
    if target_col is None:
        raise ValueError(f"No Cp target column found. Columns: {list(raw.columns)}")

    out = pd.DataFrame()
    out["compound_name"] = raw[name_col] if name_col else np.nan
    out["smiles_raw"] = raw[smiles_col]
    out["Cp_J_molK"] = pd.to_numeric(raw[target_col], errors="coerce")
    out["source"] = raw[source_col] if source_col else np.nan
    out["source_sheet"] = raw["source_sheet"]

    extra_cols = config.get("extra_feature_columns", []) or []
    if len(extra_cols) > 2:
        raise ValueError(
            f"At most 2 extra physical features are allowed, but got {len(extra_cols)}: {extra_cols}"
        )

    for col in extra_cols:
        if col not in raw.columns:
            raise ValueError(f"Extra feature column '{col}' not found in Excel columns.")
        out[col] = pd.to_numeric(raw[col], errors="coerce")

    canonical_results = out["smiles_raw"].apply(canonicalize_smiles)
    out["canonical_smiles"] = [x[0] for x in canonical_results]
    out["valid_smiles"] = [int(x[1]) for x in canonical_results]

    out = out[out["valid_smiles"].eq(1)].copy()
    out = out.dropna(subset=["Cp_J_molK"]).copy()

    return out.reset_index(drop=True)


def add_features(df: pd.DataFrame, config: Dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """
    Add SMILES-derived descriptors and optional Excel-provided physical features.
    """
    feature_set = config["feature_set"]

    X_desc = featurize_smiles_list(
        df["canonical_smiles"].tolist(),
        feature_set=feature_set,
    )

    feature_names = list(X_desc.columns)

    out = pd.concat([df.reset_index(drop=True), X_desc.reset_index(drop=True)], axis=1)

    extra_cols = config.get("extra_feature_columns", []) or []
    for col in extra_cols:
        feature_names.append(col)

    return out, feature_names


def make_training_dataset(config: Dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    raw_df = read_raw_excel(config)
    feat_df, feature_names = add_features(raw_df, config)
    return feat_df, feature_names

