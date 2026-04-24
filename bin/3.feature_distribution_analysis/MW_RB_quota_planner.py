#!/usr/bin/env python3
"""
Robust MW-RB quota planner for Cp dataset expansion.

What this script does
---------------------
1. Loads a verified dataset from Excel.
2. Builds a molecular_weight x rotatable_bonds occupancy map.
3. Marks cells eligible for external-data expansion.
4. Computes deficits relative to a target occupancy per eligible cell.
5. Allocates external-data budgets (e.g. +150, +200) across bins.
6. Saves CSV tables, per-compound bin assignments, summary text, and heatmaps.

Why this version is more robust
-------------------------------
- Validates required columns.
- Handles datasets with or without a units row under the header.
- Handles zero-deficit situations safely.
- Caps quotas so they never exceed the deficit of a bin.
- Supports multiple eligibility rules.
- Writes more audit-friendly outputs.

-------------------------------
example usage:
python MW_RB_quota_planner.py \
  --data-file mid_project_verified.xlsx \
  --sheet-name Sheet1 \
  --out-dir MW_RB_PLANNER \
  --target-per-cell 12 \
  --quota-totals 150 200 \
  --eligibility-mode rook
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MW_BINS = [0, 100, 150, 200, 260, 350, np.inf]
DEFAULT_MW_LABELS = ["<100", "100-150", "150-200", "200-260", "260-350", ">=350"]

DEFAULT_RB_BINS = [-0.1, 1, 4, 7, 11, np.inf]
DEFAULT_RB_LABELS = ["0-1", "2-4", "5-7", "8-11", "12+"]

ELIGIBILITY_MODES = {"occupied_only", "rook", "queen", "all"}
TEXT_COLOR_MODES = {"auto", "red", "white", "black"}


@dataclass(frozen=True)
class PlannerConfig:
    data_file: Path
    sheet_name: str | None
    mw_col: str
    rb_col: str
    out_dir: Path
    target_per_cell: int
    quota_totals: list[int]
    eligibility_mode: str
    detect_units_row: bool
    save_compound_bins: bool
    text_color_mode: str
    text_fontsize: int
    text_fontweight: str
    figure_dpi: int


def parse_args() -> PlannerConfig:
    p = argparse.ArgumentParser(description="Build MW-RB occupancy and quota planning tables.")
    p.add_argument("--data-file", type=Path, default=Path("mid_project_verified.xlsx"))
    p.add_argument("--sheet-name", type=str, default=None)
    p.add_argument("--mw-col", type=str, default="molecular_weight")
    p.add_argument("--rb-col", type=str, default="rotatable_bonds")
    p.add_argument("--out-dir", type=Path, default=Path("MW_RB_PLANNER"))
    p.add_argument("--target-per-cell", type=int, default=12)
    p.add_argument("--quota-totals", nargs="*", type=int, default=[150, 200])
    p.add_argument(
        "--eligibility-mode",
        type=str,
        default="rook",
        choices=sorted(ELIGIBILITY_MODES),
        help="occupied_only | rook | queen | all",
    )
    p.add_argument(
        "--no-detect-units-row",
        action="store_true",
        help="Disable automatic dropping of a units row under the header.",
    )
    p.add_argument(
        "--no-save-compound-bins",
        action="store_true",
        help="Do not save per-compound assigned MW/RB bins.",
    )
    p.add_argument(
        "--text-color-mode",
        type=str,
        default="auto",
        choices=sorted(TEXT_COLOR_MODES),
        help="How to color text annotations in heatmaps.",
    )
    p.add_argument("--text-fontsize", type=int, default=10)
    p.add_argument("--text-fontweight", type=str, default="bold")
    p.add_argument("--figure-dpi", type=int, default=180)

    args = p.parse_args()

    quota_totals = sorted({int(x) for x in args.quota_totals if int(x) > 0})
    if not quota_totals:
        raise SystemExit("[ERROR] At least one positive quota total is required.")
    if args.target_per_cell <= 0:
        raise SystemExit("[ERROR] --target-per-cell must be positive.")

    return PlannerConfig(
        data_file=args.data_file,
        sheet_name=args.sheet_name,
        mw_col=args.mw_col,
        rb_col=args.rb_col,
        out_dir=args.out_dir,
        target_per_cell=args.target_per_cell,
        quota_totals=quota_totals,
        eligibility_mode=args.eligibility_mode,
        detect_units_row=not args.no_detect_units_row,
        save_compound_bins=not args.no_save_compound_bins,
        text_color_mode=args.text_color_mode,
        text_fontsize=args.text_fontsize,
        text_fontweight=args.text_fontweight,
        figure_dpi=args.figure_dpi,
    )


def maybe_drop_units_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some project Excel sheets have a units row directly below the header.
    Drop it heuristically if most of row 0 is non-numeric text and row 1 looks like data.
    """
    if df.empty or len(df) < 2:
        return df

    row0 = df.iloc[0]
    row1 = df.iloc[1]

    row0_numeric = pd.to_numeric(row0, errors="coerce").notna().sum()
    row1_numeric = pd.to_numeric(row1, errors="coerce").notna().sum()

    row0_textish = row0.astype(str).str.strip().replace({"nan": "", "None": ""})
    row0_text_count = row0_textish.ne("").sum()

    looks_like_units = (row0_numeric <= 1) and (row1_numeric >= 2) and (row0_text_count >= max(2, len(df.columns) // 4))
    if looks_like_units:
        return df.iloc[1:].reset_index(drop=True)
    return df


def load_df(path: Path, sheet_name: str | None, detect_units_row: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if sheet_name is None:
        df = pd.read_excel(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    df = df.replace(["-", "—", "", "NA", "N/A", "na", "n/a"], np.nan)
    if detect_units_row:
        df = maybe_drop_units_row(df)
    return df


def validate_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Required columns not found: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )


def prepare_work_df(
    df: pd.DataFrame,
    mw_col: str,
    rb_col: str,
    mw_bins: list[float],
    mw_labels: list[str],
    rb_bins: list[float],
    rb_labels: list[str],
) -> pd.DataFrame:
    work = df.copy()
    validate_columns(work, [mw_col, rb_col])

    work[mw_col] = pd.to_numeric(work[mw_col], errors="coerce")
    work[rb_col] = pd.to_numeric(work[rb_col], errors="coerce")
    work = work.dropna(subset=[mw_col, rb_col]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No rows remain after dropping NaN/non-numeric MW and RB values.")

    work["mw_bin"] = pd.cut(
        work[mw_col],
        bins=mw_bins,
        labels=mw_labels,
        right=False,
        include_lowest=True,
    )
    work["rb_bin"] = pd.cut(
        work[rb_col],
        bins=rb_bins,
        labels=rb_labels,
        right=True,
        include_lowest=True,
    )

    work = work.dropna(subset=["mw_bin", "rb_bin"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No rows fall inside the configured MW/RB bins.")

    return work


def build_occupancy_table(work: pd.DataFrame, mw_labels: list[str], rb_labels: list[str]) -> pd.DataFrame:
    occ = pd.crosstab(work["mw_bin"], work["rb_bin"]).reindex(index=mw_labels, columns=rb_labels, fill_value=0)
    return occ.astype(int)


def build_eligibility_mask(occ: pd.DataFrame, mode: str) -> np.ndarray:
    arr = occ.to_numpy()
    rows, cols = arr.shape
    occupied = arr > 0

    if mode == "occupied_only":
        return occupied.copy()
    if mode == "all":
        return np.ones_like(arr, dtype=bool)

    eligible = occupied.copy()
    if mode == "rook":
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif mode == "queen":
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        raise ValueError(f"Unknown eligibility mode: {mode}")

    for i in range(rows):
        for j in range(cols):
            if occupied[i, j]:
                continue
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and occupied[ni, nj]:
                    eligible[i, j] = True
                    break

    return eligible


def build_deficit_table(occ: pd.DataFrame, eligible: np.ndarray, target_per_cell: int) -> pd.DataFrame:
    deficit = np.where(eligible, np.maximum(0, target_per_cell - occ.to_numpy()), 0)
    return pd.DataFrame(deficit.astype(int), index=occ.index, columns=occ.columns)


def allocate_quota_capped(deficit_df: pd.DataFrame, total_add: int) -> pd.DataFrame:
    """
    Allocate quota proportionally to deficit, but never exceed each bin's deficit.
    If total_add exceeds total deficit, all deficits are filled and the remainder is unused.
    """
    deficit = deficit_df.to_numpy(dtype=int)
    total_deficit = int(deficit.sum())

    quota = np.zeros_like(deficit, dtype=int)
    if total_add <= 0 or total_deficit <= 0:
        return pd.DataFrame(quota, index=deficit_df.index, columns=deficit_df.columns)

    budget = min(int(total_add), total_deficit)
    remaining = deficit.copy()

    while budget > 0:
        positive_mask = remaining > 0
        if not positive_mask.any():
            break

        weights = remaining.astype(float)
        weights[~positive_mask] = 0.0
        weights_sum = weights.sum()
        if weights_sum <= 0:
            break

        scaled = weights / weights_sum * budget
        step = np.floor(scaled).astype(int)
        step = np.minimum(step, remaining)

        if step.sum() == 0:
            flat_candidates = []
            frac = scaled - np.floor(scaled)
            pos_indices = np.argwhere(positive_mask)
            for i, j in pos_indices:
                flat_candidates.append((frac[i, j], remaining[i, j], -i, -j, i, j))
            flat_candidates.sort(reverse=True)
            for _, _, _, _, i, j in flat_candidates[:budget]:
                step[i, j] += 1

        quota += step
        remaining -= step
        spent = int(step.sum())
        if spent <= 0:
            break
        budget -= spent

    return pd.DataFrame(quota.astype(int), index=deficit_df.index, columns=deficit_df.columns)


def pick_text_color(value: float, vmax: float, mode: str) -> str:
    if mode != "auto":
        return mode
    if vmax <= 0:
        return "black"
    return "white" if value >= 0.55 * vmax else "black"


def save_heatmap(
    table: pd.DataFrame,
    title: str,
    out_png: Path,
    text_color_mode: str,
    text_fontsize: int,
    text_fontweight: str,
    figure_dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    im = ax.imshow(table.to_numpy())
    ax.set_xticks(range(len(table.columns)), labels=table.columns)
    ax.set_yticks(range(len(table.index)), labels=table.index)
    ax.set_xlabel("rotatable_bonds bin")
    ax.set_ylabel("molecular_weight bin")
    ax.set_title(title)

    vmax = float(np.nanmax(table.to_numpy())) if table.size else 0.0
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = int(table.iloc[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="red",
                fontweight=text_fontweight,
                fontsize=text_fontsize,
            )

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=figure_dpi)
    plt.close(fig)


def save_summary(
    out_path: Path,
    work: pd.DataFrame,
    occ: pd.DataFrame,
    eligible_df: pd.DataFrame,
    deficit_df: pd.DataFrame,
    quota_tables: dict[int, pd.DataFrame],
    cfg: PlannerConfig,
) -> None:
    total_rows = len(work)
    occupied_cells = int((occ.to_numpy() > 0).sum())
    eligible_cells = int(eligible_df.to_numpy().sum())
    total_deficit = int(deficit_df.to_numpy().sum())

    lines = []
    lines.append("MW-RB QUOTA PLANNER SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Input file           : {cfg.data_file}")
    lines.append(f"Sheet name           : {cfg.sheet_name}")
    lines.append(f"Rows used            : {total_rows}")
    lines.append(f"MW column            : {cfg.mw_col}")
    lines.append(f"RB column            : {cfg.rb_col}")
    lines.append(f"Eligibility mode     : {cfg.eligibility_mode}")
    lines.append(f"Target per cell      : {cfg.target_per_cell}")
    lines.append(f"Occupied cells       : {occupied_cells}")
    lines.append(f"Eligible cells       : {eligible_cells}")
    lines.append(f"Total deficit        : {total_deficit}")
    lines.append("")

    top_sparse = (
        deficit_df.stack()
        .reset_index(name="deficit")
        .rename(columns={"level_0": "mw_bin", "level_1": "rb_bin"})
        .sort_values(["deficit", "mw_bin", "rb_bin"], ascending=[False, True, True])
    )
    top_sparse = top_sparse[top_sparse["deficit"] > 0].head(10)

    lines.append("Top sparse eligible bins")
    lines.append("-" * 80)
    if top_sparse.empty:
        lines.append("None")
    else:
        for _, row in top_sparse.iterrows():
            current_count = int(occ.loc[row["mw_bin"], row["rb_bin"]])
            lines.append(
                f"{row['mw_bin']:>8} | {row['rb_bin']:<5} | current={current_count:>2d} | deficit={int(row['deficit']):>2d}"
            )
    lines.append("")

    for total_add, quota_df in quota_tables.items():
        allocated = int(quota_df.to_numpy().sum())
        lines.append(f"Quota plan for +{total_add}")
        lines.append("-" * 80)
        lines.append(f"Requested budget     : {total_add}")
        lines.append(f"Allocated budget     : {allocated}")
        if total_add > total_deficit:
            lines.append(f"Unused budget        : {total_add - allocated}")
        top_bins = (
            quota_df.stack()
            .reset_index(name="quota")
            .rename(columns={"level_0": "mw_bin", "level_1": "rb_bin"})
            .sort_values(["quota", "mw_bin", "rb_bin"], ascending=[False, True, True])
        )
        top_bins = top_bins[top_bins["quota"] > 0].head(10)
        if top_bins.empty:
            lines.append("No quota allocated.")
        else:
            for _, row in top_bins.iterrows():
                lines.append(
                    f"{row['mw_bin']:>8} | {row['rb_bin']:<5} | quota={int(row['quota']):>2d}"
                )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(cfg.data_file, cfg.sheet_name, cfg.detect_units_row)
    work = prepare_work_df(
        df=df,
        mw_col=cfg.mw_col,
        rb_col=cfg.rb_col,
        mw_bins=DEFAULT_MW_BINS,
        mw_labels=DEFAULT_MW_LABELS,
        rb_bins=DEFAULT_RB_BINS,
        rb_labels=DEFAULT_RB_LABELS,
    )

    occ = build_occupancy_table(work, DEFAULT_MW_LABELS, DEFAULT_RB_LABELS)
    eligible = build_eligibility_mask(occ, cfg.eligibility_mode)
    eligible_df = pd.DataFrame(eligible.astype(int), index=occ.index, columns=occ.columns)
    deficit_df = build_deficit_table(occ, eligible, cfg.target_per_cell)

    # Save core tables.
    occ.to_csv(cfg.out_dir / "mw_rb_occupancy_map.csv")
    eligible_df.to_csv(cfg.out_dir / "mw_rb_eligibility_map.csv")
    deficit_df.to_csv(cfg.out_dir / "mw_rb_deficit_map.csv")

    if cfg.save_compound_bins:
        work.to_csv(cfg.out_dir / "mw_rb_compound_bin_assignments.csv", index=False)

    save_heatmap(
        table=occ,
        title="MW-RB occupancy map (current dataset)",
        out_png=cfg.out_dir / "mw_rb_occupancy_map.png",
        text_color_mode=cfg.text_color_mode,
        text_fontsize=cfg.text_fontsize,
        text_fontweight=cfg.text_fontweight,
        figure_dpi=cfg.figure_dpi,
    )
    save_heatmap(
        table=eligible_df,
        title=f"MW-RB eligible cells ({cfg.eligibility_mode})",
        out_png=cfg.out_dir / "mw_rb_eligibility_map.png",
        text_color_mode=cfg.text_color_mode,
        text_fontsize=cfg.text_fontsize,
        text_fontweight=cfg.text_fontweight,
        figure_dpi=cfg.figure_dpi,
    )
    save_heatmap(
        table=deficit_df,
        title=f"MW-RB deficit map (target={cfg.target_per_cell})",
        out_png=cfg.out_dir / "mw_rb_deficit_map.png",
        text_color_mode=cfg.text_color_mode,
        text_fontsize=cfg.text_fontsize,
        text_fontweight=cfg.text_fontweight,
        figure_dpi=cfg.figure_dpi,
    )

    long_rows: list[dict[str, object]] = []
    quota_tables: dict[int, pd.DataFrame] = {}
    for total_add in cfg.quota_totals:
        quota_df = allocate_quota_capped(deficit_df, total_add)
        quota_tables[total_add] = quota_df
        quota_df.to_csv(cfg.out_dir / f"mw_rb_quota_table_plus{total_add}.csv")
        save_heatmap(
            table=quota_df,
            title=f"Suggested external-data quotas (+{total_add})",
            out_png=cfg.out_dir / f"mw_rb_quota_plus{total_add}.png",
            text_color_mode=cfg.text_color_mode,
            text_fontsize=cfg.text_fontsize,
            text_fontweight=cfg.text_fontweight,
            figure_dpi=cfg.figure_dpi,
        )

    for i, mw in enumerate(occ.index):
        for j, rb in enumerate(occ.columns):
            row = {
                "mw_bin": mw,
                "rb_bin": rb,
                "current_count": int(occ.iloc[i, j]),
                "eligible_for_sampling": bool(eligible[i, j]),
                "deficit_to_target": int(deficit_df.iloc[i, j]),
            }
            for total_add, quota_df in quota_tables.items():
                row[f"quota_if_add{total_add}"] = int(quota_df.iloc[i, j])
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(cfg.out_dir / "mw_rb_quota_long.csv", index=False)

    save_summary(
        out_path=cfg.out_dir / "mw_rb_planner_summary.txt",
        work=work,
        occ=occ,
        eligible_df=eligible_df,
        deficit_df=deficit_df,
        quota_tables=quota_tables,
        cfg=cfg,
    )

    print("Saved outputs to:")
    print(cfg.out_dir.resolve())


if __name__ == "__main__":
    main()

