#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

SIF="cp_predictor_runtime.sif"
CONFIG="configs/inference.yaml"
INPUT_FILE="data/external/test_all.xlsx"
OUTPUT_DIR="output/external_test_sif"

mkdir -p "${OUTPUT_DIR}"

apptainer run \
  --bind "${PROJECT_ROOT}/output:/app/output" \
  --bind "${PROJECT_ROOT}/data:/app/data" \
  "${SIF}" \
  --config "${CONFIG}" \
  --input-file "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Done."
echo "Results saved to: ${OUTPUT_DIR}"
