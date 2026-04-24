# Feature Verification Pipeline

## Overview
Verifies and augments chemical data using PubChem API.

## Features
- Molecular Weight
- Rotatable Bonds
- Melting / Boiling Point
- Critical Temperature

## Usage
```bash
python 5_features_DATA_editter.py
```

## Input
- mid_project_sample_data.xlsx (must contain `name` column)

## Output
- mid_project_5features_verified.xlsx

## Notes
- Auto-fills missing values
- Corrects out-of-range values (tolerance-based)
- Some compounds may not have available data

## Dependencies
```bash
pip install pandas numpy requests
```

