from __future__ import annotations

from pathlib import Path

import pandas as pd

from .summary_writer import compound_summary, overall_summary, phase_summary, summarize_numeric


def apply_clean_filters(df: pd.DataFrame, clean_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    logs = []
    initial_rows = len(work)

    numeric_cols = [
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
    ]
    for c in numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    def log_step(step_name: str, before: int, after: int):
        logs.append({
            "step": step_name,
            "rows_before": int(before),
            "rows_after": int(after),
            "rows_removed": int(before - after),
        })

    before = len(work)
    work = work[work["smiles"].fillna("").astype(str).str.strip() != ""].copy()
    log_step("require_smiles", before, len(work))

    before = len(work)
    work = work[work["heat_capacity"].notna() & (work["heat_capacity"] > 0)].copy()
    log_step("require_positive_heat_capacity", before, len(work))

    before = len(work)
    work = work[work["temperature_K"].notna() & (work["temperature_K"] > 0)].copy()
    log_step("require_positive_temperature", before, len(work))

    allowed_phases = {str(x).strip().lower() for x in clean_cfg.get("allowed_phases", ["solid", "liquid", "gas", "glass", "fluid"])}
    before = len(work)
    work = work[work["property_phase_norm"].fillna("").str.lower().isin(allowed_phases)].copy()
    log_step("allowed_property_phases", before, len(work))

    tmin = clean_cfg.get("temperature_min_K", 50.0)
    tmax = clean_cfg.get("temperature_max_K", 1500.0)
    before = len(work)
    work = work[(work["temperature_K"] >= tmin) & (work["temperature_K"] <= tmax)].copy()
    log_step(f"temperature_range_{tmin}_{tmax}", before, len(work))

    keep_nan_pressure = bool(clean_cfg.get("keep_missing_pressure", True))
    pmin = clean_cfg.get("pressure_min_kPa", 1.0)
    pmax = clean_cfg.get("pressure_max_kPa", 20000.0)
    before = len(work)
    if keep_nan_pressure:
        mask = work["pressure_kPa"].isna() | ((work["pressure_kPa"] >= pmin) & (work["pressure_kPa"] <= pmax))
        work = work[mask].copy()
    else:
        work = work[
            work["pressure_kPa"].notna()
            & (work["pressure_kPa"] >= pmin)
            & (work["pressure_kPa"] <= pmax)
        ].copy()
    log_step(f"pressure_range_{pmin}_{pmax}_keepnan_{keep_nan_pressure}", before, len(work))

    cpmin = clean_cfg.get("heat_capacity_min", 1.0e-6)
    cpmax = clean_cfg.get("heat_capacity_max", 10000.0)
    before = len(work)
    work = work[(work["heat_capacity"] >= cpmin) & (work["heat_capacity"] <= cpmax)].copy()
    log_step(f"heat_capacity_range_{cpmin}_{cpmax}", before, len(work))

    require_mw = bool(clean_cfg.get("require_molecular_weight", True))
    if require_mw:
        before = len(work)
        work = work[work["molecular_weight"].notna() & (work["molecular_weight"] > 0)].copy()
        log_step("require_molecular_weight", before, len(work))

    dedup_cols = clean_cfg.get(
        "exact_dedup_columns",
        ["structure_key", "property_phase_norm", "temperature_K", "pressure_kPa", "heat_capacity", "source_file"],
    )
    dedup_cols = [c for c in dedup_cols if c in work.columns]
    if dedup_cols:
        before = len(work)
        work = work.drop_duplicates(subset=dedup_cols, keep="first").copy()
        log_step("drop_exact_duplicates", before, len(work))

    logs.append({
        "step": "final",
        "rows_before": int(initial_rows),
        "rows_after": int(len(work)),
        "rows_removed": int(initial_rows - len(work)),
    })

    return work.reset_index(drop=True), pd.DataFrame(logs)


def build_clean_master(raw_df: pd.DataFrame, cfg: dict, paths: dict[str, Path]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("Cleaning Cp master dataset")
    print("=" * 80)

    clean_cfg = cfg.get("clean_master", {})
    clean_df, filter_log_df = apply_clean_filters(raw_df, clean_cfg)
    clean_dir = paths["clean_dir"]

    clean_df.to_csv(clean_dir / "cp_master_clean.csv", index=False, encoding="utf-8-sig")
    clean_df.to_excel(clean_dir / "cp_master_clean.xlsx", index=False)
    filter_log_df.to_csv(clean_dir / "cp_master_clean_filter_log.csv", index=False, encoding="utf-8-sig")

    summarize_numeric(clean_df, clean_dir / "cp_master_clean_numeric_summary.csv")
    phase_summary(clean_df, "property_phase_norm", clean_dir / "cp_master_clean_phase_summary.csv")
    phase_summary(clean_df, "phase_coarse", clean_dir / "cp_master_clean_phase_coarse_summary.csv")
    compound_summary(clean_df, clean_dir / "cp_master_clean_compound_summary.csv")
    overall_summary(clean_df, clean_dir / "cp_master_clean_overall_summary.csv", "cp_master_clean.csv", "cp_master_clean.xlsx")

    return clean_df
