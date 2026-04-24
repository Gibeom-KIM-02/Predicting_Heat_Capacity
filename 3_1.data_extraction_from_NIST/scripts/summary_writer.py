from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .utils import normalize_text


def summarize_numeric(df: pd.DataFrame, out_csv: Path) -> None:
    cols = [
        "heat_capacity",
        "temperature_K",
        "pressure_kPa",
        "expanded_uncertainty",
        "molecular_weight",
        "rotatable_bonds",
        "H_bond_donors",
        "H_bond_acceptors",
        "TPSA",
        "logP",
        "abs_delta_T_ref",
        "abs_delta_P_ref",
        "cp_outlier_score",
        "cp_outlier_upper_bound",
    ]

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "count": int(ser.notna().sum()),
                "min": float(ser.min()) if ser.notna().any() else math.nan,
                "max": float(ser.max()) if ser.notna().any() else math.nan,
                "mean": float(ser.mean()) if ser.notna().any() else math.nan,
                "median": float(ser.median()) if ser.notna().any() else math.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def phase_summary(df: pd.DataFrame, phase_col: str, out_csv: Path) -> None:
    rows = []
    for phase, sub in df.groupby(phase_col, dropna=False):
        tser = pd.to_numeric(sub.get("temperature_K"), errors="coerce")
        pser = pd.to_numeric(sub.get("pressure_kPa"), errors="coerce")
        cpser = pd.to_numeric(sub.get("heat_capacity"), errors="coerce")

        rows.append(
            {
                "phase": phase if str(phase).strip() else "(blank)",
                "rows": int(len(sub)),
                "unique_compounds": int(sub["structure_key"].nunique()) if "structure_key" in sub.columns else math.nan,
                "unique_inchikeys": int(sub["primary_inchikey"].replace("", pd.NA).dropna().nunique()) if "primary_inchikey" in sub.columns else math.nan,
                "temperature_min_K": float(tser.min()) if tser.notna().any() else math.nan,
                "temperature_max_K": float(tser.max()) if tser.notna().any() else math.nan,
                "pressure_min_kPa": float(pser.min()) if pser.notna().any() else math.nan,
                "pressure_max_kPa": float(pser.max()) if pser.notna().any() else math.nan,
                "cp_min": float(cpser.min()) if cpser.notna().any() else math.nan,
                "cp_max": float(cpser.max()) if cpser.notna().any() else math.nan,
                "cp_median": float(cpser.median()) if cpser.notna().any() else math.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["rows", "phase"], ascending=[False, True], kind="stable")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


def compound_summary(df: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    grouped = df.groupby("structure_key", dropna=False)

    for key, sub in grouped:
        tser = pd.to_numeric(sub.get("temperature_K"), errors="coerce")
        pser = pd.to_numeric(sub.get("pressure_kPa"), errors="coerce")
        cpser = pd.to_numeric(sub.get("heat_capacity"), errors="coerce")

        phases = sorted(
            {
                normalize_text(x)
                for x in sub.get("property_phase_norm", pd.Series(dtype=object)).dropna().tolist()
                if normalize_text(x)
            }
        )
        coarse_phases = sorted(
            {
                normalize_text(x)
                for x in sub.get("phase_coarse", pd.Series(dtype=object)).dropna().tolist()
                if normalize_text(x)
            }
        )

        first = sub.iloc[0]
        rows.append(
            {
                "structure_key": key,
                "name": first.get("name", ""),
                "smiles": first.get("smiles", ""),
                "formula": first.get("formula", ""),
                "primary_inchikey": first.get("primary_inchikey", ""),
                "n_rows": int(len(sub)),
                "available_phases": "|".join(phases),
                "available_phase_coarse": "|".join(coarse_phases),
                "temperature_min_K": float(tser.min()) if tser.notna().any() else math.nan,
                "temperature_max_K": float(tser.max()) if tser.notna().any() else math.nan,
                "pressure_min_kPa": float(pser.min()) if pser.notna().any() else math.nan,
                "pressure_max_kPa": float(pser.max()) if pser.notna().any() else math.nan,
                "cp_min": float(cpser.min()) if cpser.notna().any() else math.nan,
                "cp_max": float(cpser.max()) if cpser.notna().any() else math.nan,
                "cp_mean": float(cpser.mean()) if cpser.notna().any() else math.nan,
                "cp_median": float(cpser.median()) if cpser.notna().any() else math.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["n_rows", "name"], ascending=[False, True], kind="stable")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


def overall_summary(df: pd.DataFrame, out_csv: Path, csv_name: str, xlsx_name: str) -> None:
    out = pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "unique_compounds": int(df["structure_key"].nunique()) if "structure_key" in df.columns else math.nan,
                "unique_inchikeys": int(df["primary_inchikey"].replace("", pd.NA).dropna().nunique()) if "primary_inchikey" in df.columns else math.nan,
                "phases": int(df["property_phase_norm"].replace("", pd.NA).dropna().nunique()) if "property_phase_norm" in df.columns else math.nan,
                "phase_coarse": int(df["phase_coarse"].replace("", pd.NA).dropna().nunique()) if "phase_coarse" in df.columns else math.nan,
                "csv_path": str(out_csv.parent / csv_name),
                "xlsx_path": str(out_csv.parent / xlsx_name),
            }
        ]
    )
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
