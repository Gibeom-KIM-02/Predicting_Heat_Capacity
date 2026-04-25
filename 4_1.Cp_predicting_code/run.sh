#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Cp Prediction Pipeline Runner
# Rule-compliant version:
#   No phase labels
#   No phase router
#   No phase-specific regressors
#
# Pipeline:
#   1. Feature/model search
#   2. Train final single Cp regressor
#   3. Make final report
#   4. Run SMILES/name prediction smoke tests
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

FEATURE_SEARCH_CONFIG="configs/feature_search.yaml"
FINAL_TRAIN_CONFIG="configs/final_train.yaml"
INFERENCE_CONFIG="configs/inference.yaml"

DATA_FILE="data/processed/Cp_input_data_final.xlsx"

echo "============================================================"
echo "Cp Prediction Pipeline"
echo "Project root: ${PROJECT_ROOT}"
echo "============================================================"

echo
echo "[0] Checking required files..."
if [[ ! -f "${DATA_FILE}" ]]; then
    echo "[ERROR] Missing data file: ${DATA_FILE}"
    echo "        Put your processed Excel file there or edit configs/*.yaml."
    exit 1
fi

if [[ ! -f "${FEATURE_SEARCH_CONFIG}" ]]; then
    echo "[ERROR] Missing config: ${FEATURE_SEARCH_CONFIG}"
    exit 1
fi

if [[ ! -f "${FINAL_TRAIN_CONFIG}" ]]; then
    echo "[ERROR] Missing config: ${FINAL_TRAIN_CONFIG}"
    exit 1
fi

if [[ ! -f "${INFERENCE_CONFIG}" ]]; then
    echo "[ERROR] Missing config: ${INFERENCE_CONFIG}"
    exit 1
fi

mkdir -p output/feature_search
mkdir -p output/final_model
mkdir -p output/prediction_examples
mkdir -p models/feature_search
mkdir -p models/final_model

echo "[OK] Required files exist."

echo
echo "============================================================"
echo "[1] Running feature/model search"
echo "============================================================"
python scripts/01_run_feature_search.py \
    --config "${FEATURE_SEARCH_CONFIG}"

echo
echo "============================================================"
echo "[2] Training final Cp regressor"
echo "============================================================"
python scripts/02_train_final_regressor.py \
    --config "${FINAL_TRAIN_CONFIG}"

echo
echo "============================================================"
echo "[3] Making final report"
echo "============================================================"
python scripts/03_make_final_report.py \
    --config "${FINAL_TRAIN_CONFIG}"

echo
echo "============================================================"
echo "[4] Running SMILES prediction smoke test"
echo "============================================================"
python scripts/04_predict_from_smiles.py \
    --config "${INFERENCE_CONFIG}" \
    --smiles "CC(=O)C"

echo
echo "============================================================"
echo "[5] Running compound-name prediction smoke test"
echo "============================================================"
if python -c "import pubchempy" >/dev/null 2>&1; then
    if python scripts/05_predict_from_name.py \
        --config "${INFERENCE_CONFIG}" \
        --name "acetone"; then
        echo "[OK] Name-based prediction smoke test succeeded."
    else
        echo "[WARN] Name-based prediction smoke test failed."
        echo "       This is usually caused by blocked internet access or PubChem API failure."
        echo "       The trained model is still valid because SMILES-based prediction works."
        echo "       Use scripts/04_predict_from_smiles.py when internet access is unavailable."
    fi
else
    echo "[WARN] PubChemPy is not installed. Skipping name-based prediction test."
    echo "       Install with: pip install pubchempy"
fi

echo
echo "============================================================"
echo "Pipeline finished successfully."
echo "============================================================"

echo
echo "Main outputs:"
echo "  Feature search summary:"
echo "    output/feature_search/feature_search_summary_ranked.csv"
echo
echo "  Final model summary:"
echo "    output/final_model/final_regressor_summary.csv"
echo
echo "  Final report summary:"
echo "    output/final_model/final_report_summary.csv"
echo
echo "  Prediction tables:"
echo "    output/final_model/final_regressor_oof_predictions.csv"
echo "    output/final_model/train_validation_test_predictions.csv"
echo
echo "  Error by Cp bin:"
echo "    output/final_model/oof_error_by_cp_bin.csv"
echo "    output/final_model/train_error_by_cp_bin.csv"
echo "    output/final_model/validation_error_by_cp_bin.csv"
echo "    output/final_model/test_error_by_cp_bin.csv"
echo
echo "  Parity plots:"
echo "    output/final_model/parity_oof_cv.png"
echo "    output/final_model/parity_train_set.png"
echo "    output/final_model/parity_validation_set.png"
echo "    output/final_model/parity_test_set.png"
echo "    output/final_model/parity_final_regressor.png"
echo
echo "  Saved model:"
echo "    models/final_model/cp_regressor.joblib"
echo
echo "  Feature schema:"
echo "    models/final_model/feature_schema.json"
