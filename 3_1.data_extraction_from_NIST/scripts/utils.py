from __future__ import annotations

from typing import Any

import pandas as pd


PHASE_NORMALIZATION = {
    "liquid": "liquid",
    "gas": "gas",
    "vapor": "gas",
    "vapour": "gas",
    "air at 1 atmosphere": "gas",
    "crystal": "solid",
    "solid": "solid",
    "glass": "glass",
    "fluid (supercritical or subcritical phases)": "fluid",
    "fluid": "fluid",
    "solution": "solution",
    "liquid solution": "solution",
    "gas solution": "solution",
}

OUTPUT_COLUMNS = [
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
    "property_name",
    "property_phase_raw",
    "property_phase_norm",
    "phase_coarse",
    "temperature_K",
    "pressure_kPa",
    "expanded_uncertainty",
    "doi",
    "title",
    "journal",
    "year",
    "primary_inchi",
    "primary_inchikey",
    "source_file",
    "component_regnum",
    "component_count",
    "is_mixture",
    "structure_source",
    "structure_key",
    "abs_delta_T_ref",
    "abs_delta_P_ref",
]


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).strip().split())


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def normalize_phase_label(raw_phase: str) -> str:
    raw = normalize_text(raw_phase).lower()
    if not raw:
        return ""
    if raw in PHASE_NORMALIZATION:
        return PHASE_NORMALIZATION[raw]
    for k, v in PHASE_NORMALIZATION.items():
        if k in raw:
            return v
    return raw


def build_phase_coarse(phase: Any) -> str:
    p = str(phase).strip().lower()
    if p == "solid":
        return "solid"
    if p == "liquid":
        return "liquid"
    if p == "gas":
        return "gas"
    if p in {"glass", "fluid"}:
        return "other"
    return "unknown"


def build_structure_key(name: str, formula: str, inchi: str, inchikey: str) -> str:
    inchikey = normalize_text(inchikey)
    inchi = normalize_text(inchi)
    formula = normalize_text(formula)
    name = normalize_text(name).lower()
    if inchikey:
        return f"inchikey::{inchikey}"
    if inchi:
        return f"inchi::{inchi}"
    return f"fallback::{formula}::{name}"


def is_cp_property_name(prop_name: str) -> bool:
    s = normalize_text(prop_name).lower()
    cp_keywords = [
        "heat capacity at constant pressure",
        "isobaric heat capacity",
        "molar heat capacity at constant pressure",
        "heat capacity, cp",
        "heat capacity",
        "isobaric molar heat capacity",
    ]
    return any(k in s for k in cp_keywords)


def is_cp_only_property(prop_name: str) -> bool:
    s = normalize_text(prop_name).lower()
    cp_positive = ["constant pressure", "isobaric", "cp"]
    cp_negative = ["constant volume", "cv"]
    if any(bad in s for bad in cp_negative):
        return False
    return any(good in s for good in cp_positive)
