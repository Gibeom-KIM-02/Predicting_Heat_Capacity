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
from cp_final_model.batch_prediction import predict_external_test_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--output-dir", default="output/external_test")
    args = parser.parse_args()

    config = load_config(args.config)

    config["external_test"] = {
        "input_file": args.input_file,
        "output_dir": args.output_dir,
    }

    if args.sheet_name is not None:
        config["external_test"]["sheet_name"] = args.sheet_name

    result = predict_external_test_file(config)

    print("External test prediction finished.")
    print("Predictions:", result["prediction_path"])
    print("Summary    :", result["summary_path"])

    summary = result["summary"]
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
