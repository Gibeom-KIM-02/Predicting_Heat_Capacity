#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cp_common.config import load_config
from cp_final_model.inference import predict_cp_from_smiles, save_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = predict_cp_from_smiles(args.smiles, config)

    save_path = save_prediction(
        result,
        config.get("output_dir", "output/prediction_examples"),
    )

    print("Input SMILES      :", result["input_smiles"])
    print("Canonical SMILES  :", result["canonical_smiles"])
    print("Cp prediction     :", f"{result['Cp_pred_J_molK']:.6f} J/mol*K")
    print("Model used        :", result["model_used"])
    print("Saved             :", save_path)


if __name__ == "__main__":
    main()

