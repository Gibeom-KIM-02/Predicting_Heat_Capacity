#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cp_common.config import load_config
from cp_final_model.compound_resolver import resolve_name_to_smiles
from cp_final_model.inference import predict_cp_from_smiles, save_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    smiles = resolve_name_to_smiles(args.name)
    if smiles is None:
        raise RuntimeError(
            f"Could not resolve compound name to SMILES: {args.name}\n"
            "Possible causes:\n"
            "  1. PubChemPy is not installed.\n"
            "  2. Internet/PubChem access is blocked on this server.\n"
            "  3. The compound name is ambiguous or not found.\n"
            "Use scripts/04_predict_from_smiles.py with a known SMILES instead."
        )

    result = predict_cp_from_smiles(smiles, config)
    result["input_name"] = args.name
    result["resolved_smiles"] = smiles

    save_path = save_prediction(
        result,
        config.get("output_dir", "output/prediction_examples"),
    )

    print("Input name        :", args.name)
    print("Resolved SMILES   :", smiles)
    print("Canonical SMILES  :", result["canonical_smiles"])
    print("Cp prediction     :", f"{result['Cp_pred_J_molK']:.6f} J/mol*K")
    print("Model used        :", result["model_used"])
    print("Saved             :", save_path)


if __name__ == "__main__":
    main()

