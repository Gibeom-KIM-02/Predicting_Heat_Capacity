#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cp_common.config import load_config
from cp_final_model.plotting import make_final_report_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    make_final_report_plots(config)

    summary_path = Path(config["output_dir"]) / "final_report_summary.csv"
    print("Final report saved to:", Path(config["output_dir"]).resolve())

    if summary_path.exists():
        print(pd.read_csv(summary_path).to_string(index=False))


if __name__ == "__main__":
    main()
