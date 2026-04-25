#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cp_common.config import load_config
from cp_final_model.regressor import train_final_regressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = train_final_regressor(config)

    print("Final Cp regressor trained.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

