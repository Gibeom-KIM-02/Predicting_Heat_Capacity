#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pubchempy as pcp
from rdkit import Chem


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_CSV = INPUT_DIR / "ComputedFormationEnthalpies_2022Feb.csv"
OUTPUT_ROWS_CSV = OUTPUT_DIR / "Cp298_with_SMILES_rows.csv"
OUTPUT_UNIQUE_BY_METHOD_CSV = OUTPUT_DIR / "Cp298_with_SMILES_unique_by_method.csv"
OUTPUT_UNIQUE_ALL_METHODS_CSV = OUTPUT_DIR / "Cp298_with_SMILES_unique_all_methods_mean.csv"
OUTPUT_XLSX = OUTPUT_DIR / "Cp298_with_SMILES.xlsx"

SLEEP_BETWEEN_REQUESTS = 0.2
RETRY_COUNT = 2


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_nonnull(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return s.iloc[0]


def load_source_table(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse the CSV block-by-block and attach the calculation method to each data row.

    Expected pattern inside the file:
        # MODEL: ...
        ...
        Formula,Name,Nconf,H,S,Cp,...
        data rows...
    """
    records: list[dict] = []
    current_model: str = "UNKNOWN_MODEL"
    active_header: list[str] | None = None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            row = [clean_cell(x) for x in row]
            nonempty = [x for x in row if x]

            if not nonempty:
                continue

            first = row[0]

            # Capture model metadata from comment lines.
            if first.startswith("# MODEL:"):
                current_model = first.replace("# MODEL:", "", 1).strip() or "UNKNOWN_MODEL"
                active_header = None
                continue

            # Skip separators and generic comments.
            if first.startswith("#") or first.startswith("############################################################################"):
                continue

            # Detect a data-table header row.
            if len(row) >= 6 and row[:6] == ["Formula", "Name", "Nconf", "H", "S", "Cp298"]:
                active_header = row
                continue

            # Ignore preface metadata before the actual table header appears.
            if active_header is None:
                continue

            # Build record from the currently active header.
            row_extended = row + [""] * max(0, len(active_header) - len(row))
            rec = dict(zip(active_header, row_extended))
            rec["method"] = current_model
            records.append(rec)

    if not records:
        raise ValueError("Could not find any data rows in the CSV file.")

    df = pd.DataFrame(records)
    df.columns = [clean_cell(c) for c in df.columns]

    use_cols = ["Formula", "Name", "Cp298", "method"]
    if "Comment" in df.columns:
        use_cols.append("Comment")

    missing_cols = [c for c in ["Formula", "Name", "Cp298", "method"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns are missing after parsing: {missing_cols}")

    df = df[use_cols].copy()

    rename_map = {
        "Formula": "formula",
        "Name": "chemical_compound_name",
        "Cp298": "Cp_298_J_per_molK",
        "method": "calculation_method",
    }
    if "Comment" in df.columns:
        rename_map["Comment"] = "comment"

    df = df.rename(columns=rename_map)

    df["formula"] = df["formula"].astype(str).str.strip()
    df["chemical_compound_name"] = df["chemical_compound_name"].astype(str).str.strip()
    df["calculation_method"] = df["calculation_method"].astype(str).str.strip()
    df["Cp_298_J_per_molK"] = pd.to_numeric(df["Cp_298_J_per_molK"], errors="coerce")

    if "comment" not in df.columns:
        df["comment"] = ""
    else:
        df["comment"] = df["comment"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["chemical_compound_name", "Cp_298_J_per_molK"]).reset_index(drop=True)
    return df


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def build_query_candidates(name: str, comment: str) -> list[str]:
    queries = [name]
    if comment:
        queries.append(f"{name} {comment}")
    return queries


def query_pubchem_smiles(
    name: str,
    formula: str = "",
    comment: str = ""
) -> Tuple[Optional[str], Optional[int], Optional[str], str]:
    """
    Return:
        smiles, cid, matched_name, status
    """
    queries = build_query_candidates(name, comment)
    last_error = None

    for query in queries:
        for _ in range(RETRY_COUNT):
            try:
                cids = pcp.get_cids(query, "name")
                if not cids:
                    continue

                cid = cids[0]

                props_list = pcp.get_properties(
                    ["SMILES", "ConnectivitySMILES", "IUPACName", "MolecularFormula"],
                    cid,
                    "cid",
                )

                if not props_list:
                    return None, cid, None, "cid_found_but_no_properties"

                props = props_list[0]

                raw_smiles = props.get("SMILES") or props.get("ConnectivitySMILES")
                matched_name = props.get("IUPACName")
                matched_formula = props.get("MolecularFormula")

                canon_smiles = canonicalize_smiles(raw_smiles)

                if canon_smiles:
                    if formula and matched_formula and formula != matched_formula:
                        return canon_smiles, cid, matched_name, f"ok_formula_mismatch:{matched_formula}"
                    return canon_smiles, cid, matched_name, "ok"

                return None, cid, matched_name, "property_found_but_smiles_empty"

            except Exception as e:
                last_error = repr(e)
                time.sleep(1.0)

    return None, None, None, f"failed:{last_error}"


def add_smiles_columns(df: pd.DataFrame) -> pd.DataFrame:
    cache: dict[tuple[str, str, str], Tuple[Optional[str], Optional[int], Optional[str], str]] = {}

    smiles_col = []
    cid_col = []
    matched_name_col = []
    status_col = []

    total = len(df)

    for i, row in df.iterrows():
        # PubChem lookup is independent of calculation method,
        # so method is intentionally excluded from the cache key.
        key = (row["chemical_compound_name"], row["formula"], row["comment"])

        if key not in cache:
            cache[key] = query_pubchem_smiles(
                name=row["chemical_compound_name"],
                formula=row["formula"],
                comment=row["comment"],
            )
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        smiles, cid, matched_name, status = cache[key]

        smiles_col.append(smiles)
        cid_col.append(cid)
        matched_name_col.append(matched_name)
        status_col.append(status)

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(
                f"[{i+1}/{total}] processed | "
                f"latest={row['chemical_compound_name']} | "
                f"method={row['calculation_method']} | "
                f"status={status}"
            )

    out = df.copy()
    out["smiles"] = smiles_col
    out["pubchem_cid"] = cid_col
    out["pubchem_matched_name"] = matched_name_col
    out["lookup_status"] = status_col
    out["smiles_found"] = out["smiles"].notna()
    return out


def build_unique_mean_tables(df_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Preserve method separation.
    df_unique_by_method = (
        df_rows.groupby(
            ["chemical_compound_name", "formula", "calculation_method"],
            as_index=False,
            dropna=False,
        )
        .agg(
            Cp_298_J_per_molK=("Cp_298_J_per_molK", "mean"),
            smiles=("smiles", first_nonnull),
            pubchem_cid=("pubchem_cid", first_nonnull),
            pubchem_matched_name=("pubchem_matched_name", first_nonnull),
            lookup_status=("lookup_status", first_nonnull),
            row_count=("chemical_compound_name", "size"),
        )
        .sort_values(["chemical_compound_name", "calculation_method"])
        .reset_index(drop=True)
    )

    # Optional cross-method summary so you can still compare all-method averages.
    df_unique_all_methods = (
        df_rows.groupby(
            ["chemical_compound_name", "formula"],
            as_index=False,
            dropna=False,
        )
        .agg(
            Cp_298_J_per_molK=("Cp_298_J_per_molK", "mean"),
            smiles=("smiles", first_nonnull),
            pubchem_cid=("pubchem_cid", first_nonnull),
            pubchem_matched_name=("pubchem_matched_name", first_nonnull),
            n_methods=("calculation_method", "nunique"),
            methods=("calculation_method", lambda s: " | ".join(sorted({str(x) for x in s if pd.notna(x) and str(x).strip()}))),
            row_count=("chemical_compound_name", "size"),
        )
        .sort_values(["chemical_compound_name"])
        .reset_index(drop=True)
    )

    return df_unique_by_method, df_unique_all_methods


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_source_table(INPUT_CSV)
    print(f"Loaded rows: {len(df)}")
    print(f"Detected methods: {df['calculation_method'].nunique()}")

    df_rows = add_smiles_columns(df)
    df_unique_by_method, df_unique_all_methods = build_unique_mean_tables(df_rows)

    df_rows.to_csv(OUTPUT_ROWS_CSV, index=False)
    df_unique_by_method.to_csv(OUTPUT_UNIQUE_BY_METHOD_CSV, index=False)
    df_unique_all_methods.to_csv(OUTPUT_UNIQUE_ALL_METHODS_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_rows.to_excel(writer, index=False, sheet_name="Cp298_rows")
        df_unique_by_method.to_excel(writer, index=False, sheet_name="unique_by_method")
        df_unique_all_methods.to_excel(writer, index=False, sheet_name="unique_all_methods")

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Rows with SMILES   :", int(df_rows["smiles_found"].sum()))
    print("Rows without SMILES:", int((~df_rows["smiles_found"]).sum()))
    print("\nMethod counts:")
    print(df_rows["calculation_method"].value_counts(dropna=False).head(20))
    print("\nLookup status counts:")
    print(df_rows["lookup_status"].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()
