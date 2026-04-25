#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cp_common.config import load_config
from cp_feature_search.experiment import run_feature_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feature_search.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_feature_search(config)

    print("Feature search finished.")
    print("Summary:", result["summary_path"])
    print("Number of experiments:", result["n_experiments"])

    if result["best"] is not None:
        print("\nBest result:")
        for key, value in result["best"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
