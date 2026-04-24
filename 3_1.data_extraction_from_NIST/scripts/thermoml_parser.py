from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .structure_features import compute_rdkit_features, mol_from_compound
from .summary_writer import overall_summary, phase_summary, summarize_numeric
from .utils import (
    OUTPUT_COLUMNS,
    as_list,
    build_phase_coarse,
    build_structure_key,
    is_cp_property_name,
    normalize_phase_label,
    normalize_text,
    safe_float,
    safe_int,
)


def get_nested_prop_name(prop: dict[str, Any]) -> str:
    pmid = prop.get("Property-MethodID", {})
    pgroup = pmid.get("PropertyGroup", {})

    stack = [pgroup]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "ePropName" in cur and isinstance(cur["ePropName"], str):
                return cur["ePropName"].strip()
            for v in cur.values():
                if isinstance(v, dict):
                    stack.append(v)
                elif isinstance(v, list):
                    stack.extend([vv for vv in v if isinstance(vv, dict)])
    return ""


def get_prop_phase_name(prop: dict[str, Any]) -> str:
    return normalize_text(prop.get("PropPhaseID", {}).get("ePropPhase", ""))


def extract_variable_map(block: dict[str, Any]) -> dict[int, str]:
    var_map: dict[int, str] = {}
    for var in as_list(block.get("Variable")):
        n = safe_int(var.get("nVarNumber"))
        if n is None:
            continue
        vtype_obj = var.get("VariableID", {}).get("VariableType", {})
        if isinstance(vtype_obj, dict):
            for vv in vtype_obj.values():
                if isinstance(vv, str):
                    var_map[n] = vv
                    break
    return var_map


def extract_numvalue_conditions(
    numvalue: dict[str, Any],
    var_map: dict[int, str],
) -> tuple[float | None, float | None]:
    temperature_K = None
    pressure_kPa = None

    for vv in as_list(numvalue.get("VariableValue")):
        nvar = safe_int(vv.get("nVarNumber"))
        if nvar is None:
            continue

        var_type = var_map.get(nvar, "")
        var_value = safe_float(vv.get("nVarValue"))
        if var_value is None:
            continue

        if "Temperature" in var_type:
            temperature_K = var_value
        elif "Pressure" in var_type:
            pressure_kPa = var_value

    return temperature_K, pressure_kPa


def extract_block_constraints(block: dict[str, Any]) -> tuple[float | None, float | None]:
    temperature_K = None
    pressure_kPa = None

    for c in as_list(block.get("Constraint")):
        ctype_obj = c.get("ConstraintID", {}).get("ConstraintType", {})
        cval = safe_float(c.get("nConstraintValue"))
        if cval is None or not isinstance(ctype_obj, dict):
            continue

        values = list(ctype_obj.values())
        if not values:
            continue
        ctype = normalize_text(values[0])

        if "Temperature" in ctype:
            temperature_K = cval
        elif "Pressure" in ctype:
            pressure_kPa = cval

    return temperature_K, pressure_kPa


def extract_component_regnums(block: dict[str, Any]) -> list[int]:
    regnums = []
    for comp in as_list(block.get("Component")):
        reg = comp.get("RegNum", {})
        n = safe_int(reg.get("nOrgNum"))
        if n is not None:
            regnums.append(n)
    return regnums


def build_compound_map(json_obj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    cmap: dict[int, dict[str, Any]] = {}

    for comp in as_list(json_obj.get("Compound")):
        regnum = safe_int(comp.get("RegNum", {}).get("nOrgNum"))
        if regnum is None:
            continue

        names = comp.get("sCommonName", [])
        if isinstance(names, list) and names:
            primary_name = normalize_text(names[0])
        elif isinstance(names, str):
            primary_name = normalize_text(names)
        else:
            primary_name = ""

        cmap[regnum] = {
            "regnum": regnum,
            "name": primary_name,
            "formula": normalize_text(comp.get("sFormulaMolec", "")),
            "inchi": normalize_text(comp.get("sStandardInChI", "")),
            "inchikey": normalize_text(comp.get("sStandardInChIKey", "")),
        }
    return cmap


def parse_json_file(json_path: Path) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    citation = obj.get("Citation", {})
    compound_map = build_compound_map(obj)
    rows: list[dict[str, Any]] = []

    for block in as_list(obj.get("PureOrMixtureData")):
        component_regnums = extract_component_regnums(block)
        component_count = len(component_regnums)
        is_mixture = component_count != 1

        if component_count != 1:
            continue

        target_regnum = component_regnums[0]
        comp = compound_map.get(target_regnum)
        if not comp:
            continue

        var_map = extract_variable_map(block)
        block_T, block_P = extract_block_constraints(block)

        for prop in as_list(block.get("Property")):
            prop_name = get_nested_prop_name(prop)
            if not is_cp_property_name(prop_name):
                continue

            prop_number = prop.get("nPropNumber")
            raw_phase = get_prop_phase_name(prop)
            norm_phase = normalize_phase_label(raw_phase)

            for nv in as_list(block.get("NumValues")):
                T_num, P_num = extract_numvalue_conditions(nv, var_map)
                temperature_K = T_num if T_num is not None else block_T
                pressure_kPa = P_num if P_num is not None else block_P

                for pv in as_list(nv.get("PropertyValue")):
                    if pv.get("nPropNumber") != prop_number:
                        continue

                    cp_val = safe_float(pv.get("nPropValue"))
                    if cp_val is None:
                        continue

                    expanded_uncertainty = None
                    cu = pv.get("CombinedUncertainty", {})
                    if isinstance(cu, dict):
                        expanded_uncertainty = safe_float(cu.get("nCombExpandUncertValue"))

                    mol, structure_source = mol_from_compound(comp)
                    if mol is None:
                        smiles = ""
                        formula = comp["formula"]
                        mw = pd.NA
                        rb = pd.NA
                        hbd = pd.NA
                        hba = pd.NA
                        tpsa = pd.NA
                        logp = pd.NA
                    else:
                        feats = compute_rdkit_features(mol)
                        smiles = feats["smiles"]
                        formula = feats["formula"]
                        mw = feats["molecular_weight"]
                        rb = feats["rotatable_bonds"]
                        hbd = feats["H_bond_donors"]
                        hba = feats["H_bond_acceptors"]
                        tpsa = feats["TPSA"]
                        logp = feats["logP"]

                    row = {
                        "name": comp["name"],
                        "smiles": smiles,
                        "group": "Experimental",
                        "formula": formula,
                        "molecular_weight": mw,
                        "rotatable_bonds": rb,
                        "H_bond_donors": hbd,
                        "H_bond_acceptors": hba,
                        "TPSA": tpsa,
                        "logP": logp,
                        "melting_point": "-",
                        "boiling_point": "-",
                        "density": "-",
                        "vapor_pressure": "-",
                        "enthalpy_of_formation": "-",
                        "heat_capacity": cp_val,
                        "critical_temperature": "-",
                        "critical_pressure": "-",
                        "acentric_factor": "-",
                        "viscosity": "-",
                        "thermal_conductivity": "-",
                        "price": "-",
                        "property_name": prop_name,
                        "property_phase_raw": raw_phase,
                        "property_phase_norm": norm_phase,
                        "phase_coarse": build_phase_coarse(norm_phase),
                        "temperature_K": temperature_K,
                        "pressure_kPa": pressure_kPa,
                        "expanded_uncertainty": expanded_uncertainty,
                        "doi": normalize_text(citation.get("sDOI", "")),
                        "title": normalize_text(citation.get("sTitle", "")),
                        "journal": normalize_text(citation.get("sPubName", "")),
                        "year": normalize_text(citation.get("yrPubYr", "")),
                        "primary_inchi": comp["inchi"],
                        "primary_inchikey": comp["inchikey"],
                        "source_file": json_path.name,
                        "component_regnum": target_regnum,
                        "component_count": component_count,
                        "is_mixture": is_mixture,
                        "structure_source": structure_source,
                        "structure_key": build_structure_key(comp["name"], formula, comp["inchi"], comp["inchikey"]),
                        "abs_delta_T_ref": pd.NA,
                        "abs_delta_P_ref": pd.NA,
                    }
                    rows.append(row)

    return rows


def build_raw_master(cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    print("=" * 80)
    print("Building raw Cp master dataset from ThermoML")
    print("=" * 80)

    json_roots = [Path(p).resolve() for p in cfg["resolved_thermoml_json_roots"]]
    all_rows: list[dict[str, Any]] = []
    file_stats = []

    for root in json_roots:
        if not root.exists():
            print(f"[WARN] Missing JSON root: {root}")
            continue

        files = sorted(root.rglob("*.json"))
        print(f"[INFO] Scanning {root} -> {len(files)} JSON files")

        for path in files:
            try:
                rows = parse_json_file(path)
                all_rows.extend(rows)
                file_stats.append({"source_file": path.name, "rows_extracted": len(rows), "root": str(root)})
            except Exception as e:
                print(f"[WARN] Failed parsing {path}: {e}")

    if not all_rows:
        raise RuntimeError("No Cp rows were extracted.")

    df = pd.DataFrame(all_rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OUTPUT_COLUMNS]

    raw_dir = paths["raw_dir"]
    df.to_csv(raw_dir / "cp_master_all_phases.csv", index=False, encoding="utf-8-sig")
    df.to_excel(raw_dir / "cp_master_all_phases.xlsx", index=False)
    pd.DataFrame(file_stats).to_csv(raw_dir / "cp_master_file_stats.csv", index=False, encoding="utf-8-sig")

    summarize_numeric(df, raw_dir / "cp_master_numeric_summary.csv")
    phase_summary(df, "property_phase_norm", raw_dir / "cp_master_phase_summary.csv")
    overall_summary(df, raw_dir / "cp_master_overall_summary.csv", "cp_master_all_phases.csv", "cp_master_all_phases.xlsx")

    return df
