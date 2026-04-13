#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


ORIGINAL_SCHEMA = [
    "name",
    "smiles",
    "group",
    "formula",
    "molecular_weight",
    "rotatable_bonds",
    "H_bond_donors",
    "H_bond_acceptors",
    "TPSA",
    "logP",
    "melting_point",
    "boiling_point",
    "density",
    "vapor_pressure",
    "enthalpy_of_formation",
    "heat_capacity",
    "critical_temperature",
    "critical_pressure",
    "acentric_factor",
    "viscosity",
    "thermal_conductivity",
    "price",
]

META_COLUMNS = [
    "source_subset",
    "doi",
    "title",
    "journal",
    "year",
    "property_phase_norm",
    "temperature_K",
    "pressure_kPa",
    "expanded_uncertainty",
    "primary_inchi",
    "primary_inchikey",
    "source_file",
    "subset_row_rank",
    "structure_key",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def first_part(value: Any, sep: str = "|") -> str:
    if pd.isna(value):
        return ""
    s = str(value)
    parts = [p.strip() for p in s.split(sep)]
    return parts[0] if parts else ""


def drop_units_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_row = df.iloc[0].astype(str).to_dict()
    if first_row.get("name", "") == "-" and first_row.get("group", "") == "Experimental":
        return df.iloc[1:].reset_index(drop=True)
    return df


def parse_formula_elements(formula: str) -> list[str]:
    if not isinstance(formula, str):
        return []
    return re.findall(r"[A-Z][a-z]?", formula)


def build_primary_structure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "component_names",
        "component_formulas",
        "component_inchis",
        "component_inchikeys",
        "property_phase_norm",
        "doi",
        "source_file",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("")

    if "primary_name" not in out.columns:
        out["primary_name"] = out["component_names"].map(first_part)
    else:
        out["primary_name"] = out["primary_name"].fillna("").astype(str).str.strip()
        mask = out["primary_name"].eq("")
        out.loc[mask, "primary_name"] = out.loc[mask, "component_names"].map(first_part)

    if "primary_formula" not in out.columns:
        out["primary_formula"] = out["component_formulas"].map(first_part)
    else:
        out["primary_formula"] = out["primary_formula"].fillna("").astype(str).str.strip()
        mask = out["primary_formula"].eq("")
        out.loc[mask, "primary_formula"] = out.loc[mask, "component_formulas"].map(first_part)

    out["primary_inchi"] = out["component_inchis"].map(first_part)
    out["primary_inchikey"] = out["component_inchikeys"].map(first_part)

    return out


def mol_from_row(row: pd.Series):
    inchi = normalize_text(row.get("primary_inchi", ""))
    if inchi:
        mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=True)
        if mol is not None:
            return mol, "inchi"

    smiles = normalize_text(row.get("smiles", ""))
    if smiles and smiles != "-":
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol, "smiles"

    return None, "none"


def compute_rdkit_features(mol: Chem.Mol) -> dict[str, Any]:
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "H_bond_donors": int(Lipinski.NumHDonors(mol)),
        "H_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "logP": float(Crippen.MolLogP(mol)),
    }


def build_structure_key(row: pd.Series) -> str:
    inchikey = normalize_text(row.get("primary_inchikey", ""))
    if inchikey:
        return f"inchikey::{inchikey}"

    inchi = normalize_text(row.get("primary_inchi", ""))
    if inchi:
        return f"inchi::{inchi}"

    formula = normalize_text(row.get("formula", ""))
    name = normalize_text(row.get("name", "")).lower()
    return f"fallback::{formula}::{name}"


def representative_rank(row: pd.Series, target_T: float, target_P: float) -> tuple:
    T = pd.to_numeric(row.get("temperature_K"), errors="coerce")
    P = pd.to_numeric(row.get("pressure_kPa"), errors="coerce")
    U = pd.to_numeric(row.get("expanded_uncertainty"), errors="coerce")

    dt = abs(T - target_T) if pd.notna(T) else float("inf")
    dp = abs(P - target_P) if pd.notna(P) else float("inf")
    du = U if pd.notna(U) else float("inf")
    return (dt, dp, du)


def make_original_schema_frame(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["name"] = df["primary_name"]
    out["smiles"] = df["smiles"]
    out["group"] = group_label
    out["formula"] = df["formula"]
    out["molecular_weight"] = df["molecular_weight"]
    out["rotatable_bonds"] = df["rotatable_bonds"]
    out["H_bond_donors"] = df["H_bond_donors"]
    out["H_bond_acceptors"] = df["H_bond_acceptors"]
    out["TPSA"] = df["TPSA"]
    out["logP"] = df["logP"]

    for col in [
        "melting_point",
        "boiling_point",
        "density",
        "vapor_pressure",
        "enthalpy_of_formation",
        "critical_temperature",
        "critical_pressure",
        "acentric_factor",
        "viscosity",
        "thermal_conductivity",
        "price",
    ]:
        out[col] = "-"

    out["heat_capacity"] = pd.to_numeric(df["cp_J_per_K_per_mol"], errors="coerce")

    for meta in META_COLUMNS:
        out[meta] = df[meta] if meta in df.columns else ""

    return out


def summarize_columns(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "column": col,
            "count": int(ser.notna().sum()),
            "min": float(ser.min()) if ser.notna().any() else math.nan,
            "max": float(ser.max()) if ser.notna().any() else math.nan,
            "mean": float(ser.mean()) if ser.notna().any() else math.nan,
            "median": float(ser.median()) if ser.notna().any() else math.nan,
        })
    return rows


def process_subset(subset_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    name = subset_cfg["name"]
    path = Path(subset_cfg["path"]).resolve()
    target_T = float(global_cfg["representative_selection"]["target_temperature_K"])
    target_P = float(global_cfg["representative_selection"]["target_pressure_kPa"])

    print("\n" + "-" * 80)
    print(f"Processing subset: {name}")
    print("-" * 80)
    print(f"Input: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = build_primary_structure_columns(df)

    built_rows = []
    failed = 0

    for _, row in df.iterrows():
        mol, source = mol_from_row(row)
        if mol is None:
            failed += 1
            continue

        feats = compute_rdkit_features(mol)
        rec = row.to_dict()
        rec.update(feats)
        rec["structure_source"] = source
        rec["primary_name"] = normalize_text(rec.get("primary_name", ""))
        rec["primary_formula"] = normalize_text(rec.get("primary_formula", ""))
        rec["primary_inchi"] = normalize_text(rec.get("primary_inchi", ""))
        rec["primary_inchikey"] = normalize_text(rec.get("primary_inchikey", ""))
        rec["structure_key"] = build_structure_key(pd.Series({
            "primary_inchikey": rec.get("primary_inchikey", ""),
            "primary_inchi": rec.get("primary_inchi", ""),
            "formula": rec.get("formula", ""),
            "name": rec.get("primary_name", ""),
        }))
        rec["subset_row_rank"] = representative_rank(pd.Series(rec), target_T=target_T, target_P=target_P)
        built_rows.append(rec)

    built = pd.DataFrame(built_rows)
    print(f"Rows with valid structure: {len(built)}")
    print(f"Rows dropped due to RDKit failure: {failed}")

    if built.empty:
        return built, {
            "subset": name,
            "input_rows": int(len(df)),
            "valid_structure_rows": 0,
            "rows_after_union_dedup": 0,
            "unique_structure_keys": 0,
            "output_csv": "",
            "structure_failures": int(failed),
        }

    group_label = subset_cfg.get("group_label", f"NIST_XML_{name}")
    model_df = make_original_schema_frame(built, group_label=group_label)

    numeric_cols = [
        "molecular_weight",
        "rotatable_bonds",
        "H_bond_donors",
        "H_bond_acceptors",
        "TPSA",
        "logP",
        "heat_capacity",
        "temperature_K",
        "pressure_kPa",
    ]
    for col in numeric_cols:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    output_name = subset_cfg.get("output_name", f"{name}_model_ready.csv")
    output_path = output_dir / output_name
    model_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(model_df[["name", "formula", "molecular_weight", "rotatable_bonds", "heat_capacity"]].head(10).to_string(index=False))

    summary = {
        "subset": name,
        "input_rows": int(len(df)),
        "valid_structure_rows": int(len(model_df)),
        "rows_after_union_dedup": int(len(model_df)),
        "unique_structure_keys": int(model_df["structure_key"].nunique()),
        "output_csv": str(output_path),
        "structure_failures": int(failed),
    }
    return model_df, summary


def build_union(subset_frames: list[pd.DataFrame], subset_names: list[str], output_dir: Path, global_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not subset_frames:
        return pd.DataFrame(), {}

    union_cfg = global_cfg.get("union_output", {})
    enabled = bool(union_cfg.get("enabled", True))
    if not enabled:
        return pd.DataFrame(), {}

    combined = pd.concat(subset_frames, ignore_index=True)
    combined["subset_priority"] = combined["source_subset"].map({name: i for i, name in enumerate(subset_names)})
    combined = combined.sort_values(["subset_priority", "subset_row_rank", "name"], kind="stable").reset_index(drop=True)
    union = combined.drop_duplicates(subset=["structure_key"], keep="first").copy()
    union = union.drop(columns=["subset_priority"])

    output_name = union_cfg.get("output_name", "nist_model_ready_union.csv")
    output_path = output_dir / output_name
    union.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "subset": "UNION",
        "input_rows": int(len(combined)),
        "valid_structure_rows": int(len(combined)),
        "rows_after_union_dedup": int(len(union)),
        "unique_structure_keys": int(union["structure_key"].nunique()),
        "output_csv": str(output_path),
        "structure_failures": 0,
    }
    return union, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready NIST Cp datasets from filtered subsets.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config).resolve())

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_excel = Path(cfg["original_dataset"]["path"]).resolve()
    original_sheet = cfg["original_dataset"].get("sheet_name", "Sheet1")
    original_df = pd.read_excel(original_excel, sheet_name=original_sheet)
    original_df = drop_units_row(original_df)
    original_summary = pd.DataFrame(summarize_columns(original_df, [
        "molecular_weight",
        "rotatable_bonds",
        "H_bond_donors",
        "H_bond_acceptors",
        "TPSA",
        "logP",
        "heat_capacity",
    ]))
    original_summary.to_csv(output_dir / "original_dataset_numeric_summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Building model-ready datasets")
    print("=" * 80)
    print(f"Original dataset: {original_excel}")
    print(f"Output dir      : {output_dir}")

    subset_frames = []
    subset_summaries = []
    subset_names = []

    for subset_cfg in cfg["subsets"]:
        subset_name = subset_cfg["name"]
        subset_names.append(subset_name)
        model_df, summary = process_subset(subset_cfg, cfg, output_dir)
        if not model_df.empty:
            model_df["source_subset"] = subset_name
            output_path = Path(summary["output_csv"])
            model_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            subset_frames.append(model_df)
        subset_summaries.append(summary)

        if not model_df.empty:
            compare = pd.DataFrame(summarize_columns(model_df, [
                "molecular_weight",
                "rotatable_bonds",
                "H_bond_donors",
                "H_bond_acceptors",
                "TPSA",
                "logP",
                "heat_capacity",
            ]))
            compare.to_csv(output_dir / f"{subset_name}_numeric_summary.csv", index=False, encoding="utf-8-sig")

    union_df, union_summary = build_union(subset_frames, subset_names, output_dir, cfg)
    if union_summary:
        subset_summaries.append(union_summary)
        compare = pd.DataFrame(summarize_columns(union_df, [
            "molecular_weight",
            "rotatable_bonds",
            "H_bond_donors",
            "H_bond_acceptors",
            "TPSA",
            "logP",
            "heat_capacity",
        ]))
        compare.to_csv(output_dir / "UNION_numeric_summary.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(subset_summaries)
    summary_path = output_dir / cfg.get("summary_output_name", "model_ready_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Finished")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
