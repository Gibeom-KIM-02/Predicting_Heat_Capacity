#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=================================================="
echo "Running ChemEO scraping pipeline"
echo "=================================================="

python scripts/01_fetch_chemeo_async.py
python scripts/02_split_cp_by_phase.py

echo "=================================================="
echo "Pipeline completed successfully"
echo "=================================================="
