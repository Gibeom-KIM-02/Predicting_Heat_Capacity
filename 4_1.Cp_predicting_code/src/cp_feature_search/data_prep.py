from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first matching column name from candidates.
    """
    col_map = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


def _read_single_or_all_sheets(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Read raw Excel data.

    This function does NOT require phase-specific sheets.

    Supported modes:
    1. read_all_sheets: false
       - Read one sheet.
       - If sheet_name is null, read the first sheet.

    2. read_all_sheets: true
       - Read all sheets and concatenate them.
       - A source_sheet column is added.
       - This still does not assume that sheet names mean phase.
    """
    input_excel = Path(config["input_excel"])
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_excel}")

    read_all_sheets = bool(config.get("read_all_sheets", False))
    sheet_name = config.get("sheet_name", None)

    if read_all_sheets:
        sheet_dict = pd.read_excel(input_excel, sheet_name=None)
        frames = []

        for sname, df in sheet_dict.items():
            tmp = df.copy()
            tmp["source_sheet"] = str(sname)
            frames.append(tmp)

        if not frames:
            raise ValueError(f"No sheets were found in Excel file: {input_excel}")

        raw = pd.concat(frames, ignore_index=True)

    else:
        if sheet_name is None:
            raw = pd.read_excel(input_excel, sheet_name=0)
            raw["source_sheet"] = "sheet0"
        else:
            raw = pd.read_excel(input_excel, sheet_name=sheet_name)
            raw["source_sheet"] = str(sheet_name)

    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def read_raw_excel(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Read a raw Cp dataset without requiring solid/liquid/gas labels.

    Required columns:
        - SMILES or canonical SMILES
        - Cp target

    Optional columns:
        - compound name
        - source

    Output columns:
        task
        phase
        compound_name
        smiles_raw
        Cp_J_molK
        source
        source_sheet

    Notes:
        - task and phase are set to "all" by default.
        - They exist only to keep compatibility with feature search code.
        - No real phase label is used.
    """
    raw = _read_single_or_all_sheets(config)

    smiles_col = first_existing_column(raw, config["smiles_columns"])
    name_col = first_existing_column(raw, config.get("name_columns", []))
    target_col = first_existing_column(raw, config["target_columns"])
    source_col = first_existing_column(raw, config.get("source_columns", []))

    if smiles_col is None:
        raise ValueError(
            "No SMILES column found. "
            f"Candidate columns={config['smiles_columns']}. "
            f"Actual columns={list(raw.columns)}"
        )

    if target_col is None:
        raise ValueError(
            "No Cp target column found. "
            f"Candidate columns={config['target_columns']}. "
            f"Actual columns={list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["task"] = "all"
    out["phase"] = "all"  # compatibility only; not a real input feature
    out["compound_name"] = raw[name_col] if name_col else np.nan
    out["smiles_raw"] = raw[smiles_col]
    out["Cp_J_molK"] = pd.to_numeric(raw[target_col], errors="coerce")
    out["source"] = raw[source_col] if source_col else np.nan
    out["source_sheet"] = raw["source_sheet"]

    out = out.dropna(subset=["smiles_raw", "Cp_J_molK"]).copy()
    out = out.reset_index(drop=True)

    return out


def read_phase_excel(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    Older code may call read_phase_excel().
    In the compliant feature-search pipeline, this simply reads raw Excel
    without requiring phase labels.
    """
    return read_raw_excel(config)


def make_task_dataset(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Select data for a feature-search task.

    In the compliant pipeline, only task="all" should be used.

    The phase-specific options are intentionally not supported because using
    solid/liquid/gas labels for model selection can be interpreted as using
    unavailable phase information.
    """
    if task == "all":
        return df.copy()

    raise ValueError(
        f"Unsupported task={task}. "
        "For the rule-compliant pipeline, use only task='all'."
    )


def save_dataset_summary(df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Save simple counts for sanity checking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "group": "all",
            "n_rows": int(len(df)),
            "n_unique_smiles_raw": int(df["smiles_raw"].nunique()),
            "n_unique_compounds": int(df["compound_name"].nunique())
            if "compound_name" in df.columns
            else np.nan,
            "n_source_sheets": int(df["source_sheet"].nunique())
            if "source_sheet" in df.columns
            else np.nan,
        }
    ]

    if "source_sheet" in df.columns:
        for sheet_name, sub in df.groupby("source_sheet"):
            rows.append(
                {
                    "group": f"sheet:{sheet_name}",
                    "n_rows": int(len(sub)),
                    "n_unique_smiles_raw": int(sub["smiles_raw"].nunique()),
                    "n_unique_compounds": int(sub["compound_name"].nunique())
                    if "compound_name" in sub.columns
                    else np.nan,
                    "n_source_sheets": 1,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "dataset_summary.csv", index=False)