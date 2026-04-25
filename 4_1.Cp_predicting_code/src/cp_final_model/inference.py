from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .features import canonicalize_smiles, featurize_one_smiles


def load_inference_bundle(config: Dict[str, Any]) -> Dict[str, Any]:
    model_dir = Path(config["model_dir"])

    model_path = model_dir / "cp_regressor.joblib"
    schema_path = model_dir / "feature_schema.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing feature schema: {schema_path}")

    regressor = joblib.load(model_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return {
        "regressor": regressor,
        "schema": schema,
    }


def predict_cp_from_smiles(
    smiles: str,
    config: Dict[str, Any],
    extra_features: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Predict Cp directly from SMILES-derived descriptors.

    No phase classifier is used.
    """
    bundle = load_inference_bundle(config)
    regressor = bundle["regressor"]
    schema = bundle["schema"]

    canonical, valid = canonicalize_smiles(smiles)
    if not valid or canonical is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    feature_set = schema["feature_set"]
    feature_names = schema["feature_names"]
    required_extra = schema.get("extra_feature_columns", []) or []

    X = featurize_one_smiles(canonical, feature_set=feature_set)

    extra_features = extra_features or {}

    for col in required_extra:
        if col not in extra_features:
            raise ValueError(
                f"Missing required extra feature '{col}'. "
                "This model was trained with extra Excel-provided features."
            )
        X[col] = float(extra_features[col])

    X = X[feature_names]

    cp_pred = float(regressor.predict(X)[0])

    return {
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "Cp_pred_J_molK": cp_pred,
        "model_used": "cp_regressor.joblib",
        "feature_set": feature_set,
        "feature_names": feature_names,
    }


def save_prediction(result: Dict[str, Any], output_dir: str | Path) -> Path:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    path = outdir / "single_prediction.csv"

    flat = result.copy()
    flat["feature_names"] = json.dumps(flat.get("feature_names", []))

    pd.DataFrame([flat]).to_csv(path, index=False)
    return path

