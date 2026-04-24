from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .summary_writer import compound_summary, overall_summary, phase_summary, summarize_numeric
from .utils import is_cp_only_property


def flag_cp_outliers_by_phase_iqr(
    df: pd.DataFrame,
    phase_col: str = "phase_coarse",
    cp_col: str = "heat_capacity",
    min_group_size: int = 30,
    iqr_multiplier: float = 3.0,
    iqr_multiplier_by_phase: dict[str, float] | None = None,
    use_log10: bool = True,
) -> pd.DataFrame:
    work = df.copy()
    work[cp_col] = pd.to_numeric(work[cp_col], errors="coerce")

    work["cp_outlier_flag"] = False
    work["cp_outlier_score"] = pd.NA
    work["cp_outlier_upper_bound"] = pd.NA
    work["cp_outlier_phase_rule"] = ""

    for phase, idx in work.groupby(phase_col).groups.items():
        sub = work.loc[idx].copy()
        sub = sub[sub[cp_col].notna() & (sub[cp_col] > 0)].copy()

        if len(sub) < min_group_size:
            work.loc[idx, "cp_outlier_phase_rule"] = f"skip_small_group_n<{min_group_size}"
            continue

        phase_key = str(phase).strip().lower()
        phase_multiplier = iqr_multiplier
        if iqr_multiplier_by_phase and phase_key in iqr_multiplier_by_phase:
            phase_multiplier = float(iqr_multiplier_by_phase[phase_key])

        if use_log10:
            x = sub[cp_col].map(lambda v: math.log10(v))
            rule_name = f"log10_iqr_{phase_multiplier}"
        else:
            x = sub[cp_col]
            rule_name = f"raw_iqr_{phase_multiplier}"

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + phase_multiplier * iqr

        work.loc[sub.index, "cp_outlier_score"] = x.values
        work.loc[sub.index, "cp_outlier_upper_bound"] = upper
        work.loc[sub.index, "cp_outlier_phase_rule"] = rule_name

        mask = x > upper
        if mask.any():
            work.loc[sub.index[mask], "cp_outlier_flag"] = True

    return work


def apply_representative_outlier_filter(df: pd.DataFrame, repr_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = repr_cfg.get("outlier_filter", {})
    enabled = bool(cfg.get("enabled", False))

    work = df.copy()
    if not enabled:
        log_df = pd.DataFrame([
            {
                "step": "cp_outlier_filter_disabled",
                "rows_before": int(len(work)),
                "rows_after": int(len(work)),
                "rows_removed": 0,
            }
        ])
        return work, log_df

    phase_col = cfg.get("phase_column", "phase_coarse")
    cp_col = cfg.get("cp_column", "heat_capacity")
    min_group_size = int(cfg.get("min_group_size", 30))
    iqr_multiplier = float(cfg.get("iqr_multiplier", 3.0))
    iqr_multiplier_by_phase = cfg.get("iqr_multiplier_by_phase", {})
    use_log10 = bool(cfg.get("use_log10", True))
    remove_flagged = bool(cfg.get("remove_flagged", True))

    before = len(work)
    work = flag_cp_outliers_by_phase_iqr(
        work,
        phase_col=phase_col,
        cp_col=cp_col,
        min_group_size=min_group_size,
        iqr_multiplier=iqr_multiplier,
        iqr_multiplier_by_phase=iqr_multiplier_by_phase,
        use_log10=use_log10,
    )

    flagged = int(work["cp_outlier_flag"].fillna(False).sum())
    if remove_flagged:
        work = work[~work["cp_outlier_flag"].fillna(False)].copy()

    after = len(work)
    log_df = pd.DataFrame([
        {
            "step": f"cp_outlier_filter_phasewise_{phase_col}",
            "rows_before": int(before),
            "rows_after": int(after),
            "rows_removed": int(before - after),
            "flagged_rows": flagged,
            "min_group_size": min_group_size,
            "iqr_multiplier": iqr_multiplier,
            "iqr_multiplier_by_phase": str(iqr_multiplier_by_phase),
            "use_log10": use_log10,
        }
    ])
    return work, log_df


def build_temperature_window_dataset(clean_df: pd.DataFrame, cfg: dict, paths: dict[str, Path]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("Building temperature-window dataset")
    print("=" * 80)

    window_cfg = cfg.get("temperature_window_dataset", {})
    ref_T = float(window_cfg.get("reference_temperature_K", 298.15))
    delta_T = float(window_cfg.get("window_half_width_K", 30.0))
    dedup_mode = str(window_cfg.get("dedup_mode", "keep_all")).strip().lower()

    work = clean_df.copy()
    work["temperature_K"] = pd.to_numeric(work["temperature_K"], errors="coerce")
    work["abs_delta_T_ref"] = (work["temperature_K"] - ref_T).abs()

    before = len(work)
    work = work[work["abs_delta_T_ref"].notna() & (work["abs_delta_T_ref"] <= delta_T)].copy()
    after_window = len(work)

    filter_log = [{
        "step": f"temperature_window_{ref_T}_pm_{delta_T}",
        "rows_before": int(before),
        "rows_after": int(after_window),
        "rows_removed": int(before - after_window),
    }]

    if dedup_mode == "nearest_per_structure_phase":
        group_cols = ["structure_key", "phase_coarse"]
        work = work.sort_values(["abs_delta_T_ref", "temperature_K", "pressure_kPa", "source_file"], ascending=[True, True, True, True], kind="stable")
        work = work.drop_duplicates(subset=group_cols, keep="first").copy()
        filter_log.append({
            "step": "nearest_per_structure_phase",
            "rows_before": int(after_window),
            "rows_after": int(len(work)),
            "rows_removed": int(after_window - len(work)),
        })
    elif dedup_mode == "nearest_per_structure":
        group_cols = ["structure_key"]
        work = work.sort_values(["abs_delta_T_ref", "temperature_K", "pressure_kPa", "source_file"], ascending=[True, True, True, True], kind="stable")
        work = work.drop_duplicates(subset=group_cols, keep="first").copy()
        filter_log.append({
            "step": "nearest_per_structure",
            "rows_before": int(after_window),
            "rows_after": int(len(work)),
            "rows_removed": int(after_window - len(work)),
        })
    elif dedup_mode != "keep_all":
        raise ValueError(
            f"Unsupported dedup_mode: {dedup_mode}. Use one of: keep_all, nearest_per_structure_phase, nearest_per_structure"
        )

    window_dir = paths["window_dir"]
    work.to_csv(window_dir / "cp_master_298K_window.csv", index=False, encoding="utf-8-sig")
    work.to_excel(window_dir / "cp_master_298K_window.xlsx", index=False)
    pd.DataFrame(filter_log).to_csv(window_dir / "cp_master_298K_window_filter_log.csv", index=False, encoding="utf-8-sig")

    summarize_numeric(work, window_dir / "cp_master_298K_window_numeric_summary.csv")
    phase_summary(work, "property_phase_norm", window_dir / "cp_master_298K_window_phase_summary.csv")
    phase_summary(work, "phase_coarse", window_dir / "cp_master_298K_window_phase_coarse_summary.csv")
    compound_summary(work, window_dir / "cp_master_298K_window_compound_summary.csv")
    overall_summary(work, window_dir / "cp_master_298K_window_overall_summary.csv", "cp_master_298K_window.csv", "cp_master_298K_window.xlsx")

    return work


def build_representative_dataset(clean_df: pd.DataFrame, cfg: dict, paths: dict[str, Path]) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("Building representative ambient-like dataset")
    print("=" * 80)

    repr_cfg = cfg.get("representative_dataset", {})
    enabled = bool(repr_cfg.get("enabled", True))
    if not enabled:
        print("[INFO] representative_dataset.enabled = false, skipping.")
        return pd.DataFrame()

    ref_T = float(repr_cfg.get("reference_temperature_K", 298.15))
    dT = float(repr_cfg.get("temperature_window_half_width_K", 30.0))
    ref_P = float(repr_cfg.get("reference_pressure_kPa", 101.325))
    dP = float(repr_cfg.get("pressure_window_half_width_kPa", 20.0))
    keep_missing_pressure = bool(repr_cfg.get("keep_missing_pressure", False))
    grouping_columns = [c for c in repr_cfg.get("grouping_columns", ["structure_key", "phase_coarse"]) if c in clean_df.columns]

    work = clean_df.copy()
    work["temperature_K"] = pd.to_numeric(work["temperature_K"], errors="coerce")
    work["pressure_kPa"] = pd.to_numeric(work["pressure_kPa"], errors="coerce")
    work["expanded_uncertainty"] = pd.to_numeric(work["expanded_uncertainty"], errors="coerce")

    before = len(work)
    work = work[work["property_name"].fillna("").map(is_cp_only_property)].copy()
    after_cp = len(work)

    work["abs_delta_T_ref"] = (work["temperature_K"] - ref_T).abs()
    work = work[work["abs_delta_T_ref"].notna() & (work["abs_delta_T_ref"] <= dT)].copy()
    after_T = len(work)

    work["abs_delta_P_ref"] = (work["pressure_kPa"] - ref_P).abs()
    if keep_missing_pressure:
        mask_p = work["pressure_kPa"].isna() | (work["abs_delta_P_ref"] <= dP)
    else:
        mask_p = work["pressure_kPa"].notna() & (work["abs_delta_P_ref"] <= dP)
    work = work[mask_p].copy()
    after_P = len(work)

    work, outlier_log_df = apply_representative_outlier_filter(work, repr_cfg)

    work["uncertainty_rank"] = work["expanded_uncertainty"].fillna(1.0e18)
    sort_cols = [c for c in repr_cfg.get("selection_priority", ["abs_delta_T_ref", "abs_delta_P_ref", "uncertainty_rank"]) if c in work.columns]
    for c in ["temperature_K", "pressure_kPa", "source_file"]:
        if c in work.columns:
            sort_cols.append(c)
    work = work.sort_values(sort_cols, ascending=True, kind="stable")

    before_dedup = len(work)
    if grouping_columns:
        work = work.drop_duplicates(subset=grouping_columns, keep="first").copy()
    after_dedup = len(work)
    work = work.drop(columns=["uncertainty_rank"], errors="ignore")

    window_dir = paths["window_dir"]
    out_csv = window_dir / repr_cfg.get("output_csv", "cp_master_ambient_representative.csv")
    out_xlsx = window_dir / repr_cfg.get("output_xlsx", "cp_master_ambient_representative.xlsx")
    out_log = window_dir / "cp_master_ambient_representative_filter_log.csv"

    work.to_csv(out_csv, index=False, encoding="utf-8-sig")
    work.to_excel(out_xlsx, index=False)

    log_rows = [
        {"step": "cp_only_filter", "rows_before": before, "rows_after": after_cp, "rows_removed": before - after_cp},
        {"step": f"temperature_window_{ref_T}_pm_{dT}", "rows_before": after_cp, "rows_after": after_T, "rows_removed": after_cp - after_T},
        {"step": f"pressure_window_{ref_P}_pm_{dP}_keepnan_{keep_missing_pressure}", "rows_before": after_T, "rows_after": after_P, "rows_removed": after_T - after_P},
    ]
    log_rows.extend(outlier_log_df.to_dict(orient="records"))
    log_rows.append({
        "step": f"representative_dedup_by_{'|'.join(grouping_columns)}",
        "rows_before": before_dedup,
        "rows_after": after_dedup,
        "rows_removed": before_dedup - after_dedup,
    })
    pd.DataFrame(log_rows).to_csv(out_log, index=False, encoding="utf-8-sig")

    summarize_numeric(work, window_dir / "cp_master_ambient_representative_numeric_summary.csv")
    phase_summary(work, "property_phase_norm", window_dir / "cp_master_ambient_representative_phase_summary.csv")
    phase_summary(work, "phase_coarse", window_dir / "cp_master_ambient_representative_phase_coarse_summary.csv")
    compound_summary(work, window_dir / "cp_master_ambient_representative_compound_summary.csv")
    overall_summary(work, window_dir / "cp_master_ambient_representative_overall_summary.csv", out_csv.name, out_xlsx.name)

    return work
