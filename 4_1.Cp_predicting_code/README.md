# Cp Prediction Model

This project develops a machine-learning model to predict the constant-pressure heat capacity, \(C_p\), of chemical compounds in units of J/mol·K.

The final model uses compound names or SMILES strings as input. Compound names are first converted to SMILES, and molecular descriptors are generated from SMILES using RDKit. The final Cp prediction is performed using a Random Forest regression model.

No phase labels such as solid/liquid/gas are used as model inputs.

---

## 1. Project Objective

The goal of this project is to predict the constant-pressure heat capacity \(C_p\) of chemical compounds.

The model follows the assignment input rule:

- Required input: compound name or SMILES
- Additional physical features: optional, at most two columns from the provided Excel dataset
- SMILES-derived descriptors are generated using RDKit and used as molecular representation

In the final submitted model, no manually provided phase label is used. The prediction pipeline is:

```text
compound name or SMILES with Cp
        ↓
name-to-SMILES resolution, if needed
        ↓
RDKit canonical SMILES
        ↓
RDKit molecular descriptors
        ↓
RandomForestRegressor
        ↓
Predicted Cp [J/mol·K]
```

---

## 2. Final Model Summary

The final selected model is:

```text
Feature set : chem_compact_v1
Model       : RandomForestRegressor
Target      : Cp [J/mol·K]
```

The `chem_compact_v1` descriptor set was selected to balance prediction accuracy, chemical interpretability, and robustness. It uses a compact set of chemically meaningful RDKit descriptors related to molecular size, shape, polarity, and flexibility.

Main descriptors include:

```text
NumValenceElectrons
HeavyAtomCount
LabuteASA
Kappa1
NumRotatableBonds
MolLogP
TPSA
NumHAcceptors
FractionCSP3
HeteroAtomFraction
```

These descriptors are chemically reasonable for Cp prediction because molecular heat capacity is strongly related to molecular size, accessible degrees of freedom, molecular flexibility, and structural complexity.

---

## 3. Directory Structure

The project is organized into configuration files, executable scripts, reusable source modules, trained model artifacts, and output reports.

```text
4_1.Cp_prediction/
├─ configs/
│  ├─ feature_search.yaml        # Configuration for broad feature/model search
│  ├─ final_train.yaml           # Main configuration for final model training
│  └─ inference.yaml             # Configuration for SMILES/name/external prediction
│
├─ data/
│  ├─ processed/
│  │  └─ Cp_input_data_final.xlsx # Final cleaned dataset used for model training
│  └─ external/
│     └─ test_all.xlsx            # Example external test file
│
├─ scripts/
│  ├─ 01_run_feature_search.py    # Runs broad feature/model benchmarking
│  ├─ 02_train_final_regressor.py # Trains the final Cp regression model
│  ├─ 03_make_final_report.py     # Generates final report metrics and plots
│  ├─ 04_predict_from_smiles.py   # Predicts Cp from a single SMILES string
│  ├─ 05_predict_from_name.py     # Predicts Cp from a single compound name
│  ├─ 06_predict_external_test.py # Predicts Cp for an external Excel/CSV test file
│  └─ 07_sweep_final_models.py    # Optional model/feature sweep for selection evidence
│
├─ src/
│  ├─ cp_common/
│  │  ├─ __init__.py
│  │  └─ config.py                # YAML configuration loader
│  │
│  ├─ cp_feature_search/
│  │  ├─ __init__.py
│  │  ├─ data_prep.py             # Reads raw Excel data for feature search
│  │  ├─ descriptors.py           # Generates RDKit/fingerprint descriptors
│  │  ├─ evaluate.py              # Cross-validation and metric calculation
│  │  ├─ experiment.py            # Main feature/model search workflow
│  │  ├─ models.py                # Candidate regression models for benchmarking
│  │  └─ plotting.py              # Feature-search plots such as MAE ranking plots
│  │
│  └─ cp_final_model/
│     ├─ __init__.py
│     ├─ batch_prediction.py      # Batch prediction for external Excel/CSV files
│     ├─ compound_resolver.py     # Compound name → SMILES resolver using PubChem/PUG-REST
│     ├─ data_prep.py             # Final training dataset preparation
│     ├─ feature_registry.py      # Final descriptor set definitions
│     ├─ features.py              # RDKit descriptor calculation for final model
│     ├─ inference.py             # Single-compound inference utilities
│     ├─ models.py                # Final Cp regressor builder
│     ├─ plotting.py              # Final report plotting utilities
│     └─ regressor.py             # Final training, evaluation, SHAP, and model saving logic
│
├─ models/
│  └─ final_model/
│     ├─ cp_regressor.joblib      # Saved trained final model
│     └─ feature_schema.json      # Saved feature schema used by the trained model
│
├─ output/
│  ├─ feature_search/             # Outputs from broad feature/model search
│  ├─ final_model/                # Outputs from final model training and evaluation
│  └─ external_test/              # Outputs from external test prediction
│
├─ run.sh                         # Runs the full pipeline
├─ how_to_run_external.txt        # Example command for external test prediction
└─ README.md
```

---

### 3.1 Source Module Details

#### `src/cp_common/`

This folder contains shared utilities used across the project.

```text
cp_common/
├─ __init__.py
└─ config.py      # Loads YAML configuration files into Python dictionaries
```

#### `src/cp_feature_search/`

This folder is used for broad feature/model benchmarking. It is mainly used to justify the choice of molecular representation and regression model.

```text
cp_feature_search/
├─ data_prep.py   # Reads the input Excel file and standardizes column names
├─ descriptors.py # Generates descriptor sets such as basic_2, rdkit_selected, rdkit_small, Morgan, and MACCS
├─ evaluate.py    # Performs GroupKFold/KFold cross-validation and computes MAE, RMSE, R²
├─ experiment.py  # Runs the full feature_set × model search loop
├─ models.py      # Defines candidate regressors such as Ridge, ElasticNet, RF, GBR, SVR
└─ plotting.py    # Saves feature-search summary plots
```

The purpose of this module is not to train the final submitted model directly, but to answer:

```text
Which molecular representation and model class work best for Cp prediction?
```

#### `src/cp_final_model/`

This folder contains the final model pipeline used for training, saving, inference, and external prediction.

```text
cp_final_model/
├─ batch_prediction.py      # Predicts Cp for many external compounds from Excel/CSV files
├─ compound_resolver.py     # Converts compound names to SMILES using PubChem/PUG-REST
├─ data_prep.py             # Reads the final training Excel and prepares model-ready data
├─ feature_registry.py      # Defines final descriptor sets such as rdkit_small and chem_compact_v1
├─ features.py              # Calculates RDKit descriptors from canonical SMILES
├─ inference.py             # Predicts Cp for a single SMILES or name
├─ models.py                # Builds the final RandomForest/GradientBoosting regressor
├─ plotting.py              # Generates final report plots
└─ regressor.py             # Main final training workflow
```

`regressor.py` handles:

```text
1. GroupKFold OOF evaluation
2. Train/validation/test split evaluation
3. Final model fitting
4. Model saving
5. Feature importance analysis
6. Optional SHAP analysis
7. Cp-bin error analysis
```

---

### 3.2 Output Directory Details

#### `output/feature_search/`

This folder stores the broad feature/model search results.

```text
output/feature_search/
├─ dataset_summary.csv                 # Number of rows and unique compounds in the input data
├─ feature_search_summary.csv          # Raw feature/model search results
├─ feature_search_summary_ranked.csv   # Feature/model search results ranked by MAE
├─ best_feature_search_model.csv       # Best model from broad feature search
├─ top20_feature_search_mae.png        # Top-20 MAE comparison plot
├─ oof_predictions/                    # OOF prediction CSVs for each feature/model pair
└─ parity_plots/                       # Parity plots for feature/model search results
```

This directory supports the feature/model selection discussion. For example, it shows whether RDKit descriptors outperform the two-feature baseline or fingerprint-based models.

#### `output/final_model/`

This folder stores final model training results, evaluation metrics, and interpretability outputs.

```text
output/final_model/
├─ final_regressor_summary.csv             # Main summary of OOF, train, validation, and test metrics
├─ final_report_summary.csv                # Final report summary metrics
├─ final_regressor_oof_predictions.csv     # OOF predictions from GroupKFold CV
├─ train_validation_test_predictions.csv   # Predictions for train/validation/test split
│
├─ parity_oof_cv.png                       # OOF parity plot
├─ parity_train_set.png                    # Train-set parity plot
├─ parity_validation_set.png               # Validation-set parity plot
├─ parity_test_set.png                     # Test-set parity plot
├─ parity_final_regressor.png              # Final model parity plot
│
├─ oof_error_by_cp_bin.csv                 # OOF error by Cp quantile bin
├─ train_error_by_cp_bin.csv               # Train error by Cp quantile bin
├─ validation_error_by_cp_bin.csv          # Validation error by Cp quantile bin
├─ test_error_by_cp_bin.csv                # Test error by Cp quantile bin
│
├─ final_regressor_feature_importance.csv  # Feature importance from the final tree model
├─ training_logs/                          # Learning/log curves such as n_estimators vs MAE
└─ interpretability/                       # Feature importance plots and optional SHAP outputs
```

The most important files for reporting final model performance are:

```text
final_regressor_summary.csv
final_report_summary.csv
parity_oof_cv.png
parity_test_set.png
test_error_by_cp_bin.csv
```

#### `output/final_model/interpretability/`

This folder is generated when interpretability analysis is enabled in `configs/final_train.yaml`.

```text
output/final_model/interpretability/
├─ feature_importance_all.csv        # Full feature importance ranking
├─ feature_importance_top20.csv      # Top-20 feature importance table
├─ feature_importance_top20.png      # Top-20 feature importance bar plot
├─ shap_mean_abs_importance.csv      # Mean absolute SHAP values
├─ shap_values_sample.csv            # SHAP values for sampled data
├─ shap_summary_bar.png              # SHAP feature importance bar plot
└─ shap_summary_beeswarm.png         # SHAP beeswarm plot
```

These files are used to explain which molecular descriptors most strongly affect Cp prediction.

#### `output/final_model/training_logs/`

This folder stores training-log style outputs.

```text
output/final_model/training_logs/
├─ random_forest_n_estimators_curve.csv # MAE/RMSE/R² as n_estimators changes
└─ random_forest_n_estimators_curve.png # Plot of n_estimators vs train/validation/test MAE
```

For GradientBoosting models, staged prediction logs may also be saved.

#### `output/external_test/`

This folder stores predictions on external test files.

```text
output/external_test/
├─ external_test_predictions.csv # Row-by-row Cp predictions
├─ external_test_summary.csv     # External test MAE, RMSE, R²
└─ parity_external_test.png      # External test parity plot
```

`external_test_predictions.csv` includes:

```text
compound_name
input_smiles
resolved_smiles
canonical_smiles
resolve_status
Cp_true_J_molK
Cp_pred_J_molK
abs_error
error
```

The `resolve_status` column shows how the SMILES was obtained:

```text
from_smiles_column
resolved_from_local_lookup
resolved_from_manual_map
resolved_from_pubchem
name_resolution_failed
```

---

### 3.3 Model Artifact Details

The trained final model is saved in:

```text
models/final_model/
├─ cp_regressor.joblib
└─ feature_schema.json
```

`cp_regressor.joblib` contains the trained scikit-learn pipeline.

`feature_schema.json` stores the exact feature set and feature names required for inference. This ensures that external predictions use the same descriptor order as training.

---

## 4. Environment

The code was developed in a Python environment with the following main packages:

```text
pandas
numpy
scikit-learn
matplotlib
rdkit
joblib
pubchempy
requests
shap   # optional, for interpretability analysis
openpyxl
```

If needed, install missing packages using:

```bash
pip install pandas numpy scikit-learn matplotlib joblib pubchempy requests shap openpyxl
```

RDKit is usually installed with conda:

```bash
conda install -c conda-forge rdkit
```

---

## 5. Configuration Files

### 5.1 `configs/final_train.yaml`

This is the main configuration file for final model training.

Important fields:

```yaml
input_excel: data/processed/Cp_input_data_final.xlsx
output_dir: output/final_model
model_dir: models/final_model

feature_set: chem_compact_v1

split:
  enabled: true
  train_size: 0.60
  val_size: 0.20
  test_size: 0.20
  n_bins: 10
  random_seed: 2301
  refit_final_on: full

regressor:
  model_type: random_forest
  n_estimators: 800
  min_samples_leaf: 1
  min_samples_split: 2
  max_depth: null
  max_features: sqrt
  bootstrap: true
```

The model is evaluated using a train/validation/test split, and the final deployable model is refit on the full cleaned dataset when `refit_final_on: full`.

---

## 6. How to Run the Full Pipeline

From the project root directory:

```bash
bash run.sh
```

This runs:

```text
1. Feature/model search
2. Final Cp regressor training
3. Final report generation
4. SMILES-based prediction smoke test
5. Compound-name-based prediction smoke test
```

Main outputs:

```text
output/feature_search/feature_search_summary_ranked.csv
output/final_model/final_regressor_summary.csv
output/final_model/final_report_summary.csv
output/final_model/parity_oof_cv.png
output/final_model/parity_train_set.png
output/final_model/parity_validation_set.png
output/final_model/parity_test_set.png
models/final_model/cp_regressor.joblib
models/final_model/feature_schema.json
```

---

## 7. Step-by-Step Execution

### 7.1 Feature/model search

```bash
python scripts/01_run_feature_search.py \
  --config configs/feature_search.yaml
```

This compares multiple molecular representations and regression models.

Feature sets include:

```text
basic_2
rdkit_selected
rdkit_small
morgan
maccs
```

Models include:

```text
linear
ridge
elasticnet
random_forest
gradient_boosting
svr
```

The broad feature search showed that RDKit descriptor sets with Random Forest consistently outperformed the two-feature baseline and fingerprint-based models.

---

### 7.2 Final model training

```bash
python scripts/02_train_final_regressor.py \
  --config configs/final_train.yaml
```

This trains the final Cp regression model and saves:

```text
models/final_model/cp_regressor.joblib
models/final_model/feature_schema.json
output/final_model/final_regressor_summary.csv
output/final_model/final_regressor_oof_predictions.csv
output/final_model/train_validation_test_predictions.csv
```

The saved model can later be used for external test prediction without retraining.

---

### 7.3 Final report generation

```bash
python scripts/03_make_final_report.py \
  --config configs/final_train.yaml
```

This generates final summary metrics and parity plots.

Important outputs:

```text
output/final_model/final_report_summary.csv
output/final_model/parity_final_regressor.png
```

---

## 8. Prediction from a Single SMILES

Example:

```bash
python scripts/04_predict_from_smiles.py \
  --config configs/inference.yaml \
  --smiles "CC(=O)C"
```

Output includes:

```text
Input SMILES
Canonical SMILES
Predicted Cp [J/mol·K]
```

---

## 9. Prediction from a Compound Name

Example:

```bash
python scripts/05_predict_from_name.py \
  --config configs/inference.yaml \
  --name "acetone"
```

The compound name is resolved to SMILES using PubChem/PUG-REST, then canonicalized with RDKit before descriptor calculation.

---

## 10. External Test Prediction

External test files may contain either SMILES or compound names.

Example:

```bash
python scripts/06_predict_external_test.py \
  --config configs/inference.yaml \
  --input-file data/external/test_all.xlsx \
  --output-dir output/external_test
```

The external test file may have columns such as:

```text
name
smiles
heat_capacity
```

or:

```text
chemical_compound_name
canonical_smiles
Cp [J/mol*K]
```

If true Cp values are available, the script calculates:

```text
MAE
RMSE
R²
```

Outputs:

```text
output/external_test/external_test_predictions.csv
output/external_test/external_test_summary.csv
output/external_test/parity_external_test.png
```

---

## 11. Name-to-SMILES Resolution

For name-only external test files, the pipeline resolves names to SMILES using the following priority:

```text
1. Local name-SMILES lookup from data/processed/Cp_input_data_final.xlsx
2. Manual correction map, if provided
3. PubChem/PUG-REST online search
4. Failed row saved for inspection
```

This improves robustness when compound names are provided without SMILES.

The relevant configuration is in `configs/inference.yaml`:

```yaml
name_resolution:
  use_local_lookup: true
  local_lookup_files:
    - data/processed/Cp_input_data_final.xlsx
  sheet_name: null
  allow_pubchem_fallback: true
  manual_map_file: data/external/name_smiles_manual_map.csv
```

The manual map is optional. If needed, create:

```text
data/external/name_smiles_manual_map.csv
```

with the format:

```csv
compound_name,smiles
example compound,CCO
```

---

## 12. Model/Feature Selection Evidence

### 12.1 Broad feature search

The broad feature search compared:

```text
basic_2
rdkit_selected
rdkit_small
morgan
maccs
```

with multiple regression models. The result showed that RDKit descriptor sets with Random Forest performed best.

This supports the choice of RDKit molecular descriptors as the main molecular representation.

### 12.2 Final model sweep

An additional final-model sweep was used during development to compare final descriptor/model choices using the same train/validation/test split.

Example command:

```bash
python scripts/07_sweep_final_models.py \
  --config configs/final_train.yaml \
  --out-dir output/final_model_sweep_compact
```

This sweep compared models such as:

```text
rdkit_small + RandomForest
selected_rdkit_v1 + RandomForest
chem_compact_v1 + RandomForest
SHAP-top descriptor subsets
GradientBoostingRegressor
sample-weighted RandomForest
```

The sweep showed that `rdkit_small + RandomForest` achieved the lowest test MAE, while `chem_compact_v1 + RandomForest` achieved nearly identical validation/test performance with a smaller and more interpretable descriptor set.

Therefore, the final submitted model uses:

```text
chem_compact_v1 + RandomForestRegressor
```

to balance accuracy, interpretability, and robustness.

The full sweep output does not need to be included in the final submission directory, but the script is kept for reproducibility.

---

## 13. Interpretability Analysis

If enabled in `configs/final_train.yaml`, interpretability outputs are generated automatically during final model training.

Configuration:

```yaml
interpretability:
  enabled: true
  output_dir: output/final_model/interpretability
  max_samples: 500
  random_seed: 6739
  save_shap: true
  save_feature_importance_plot: true
```

Outputs:

```text
output/final_model/interpretability/feature_importance_all.csv
output/final_model/interpretability/feature_importance_top20.csv
output/final_model/interpretability/feature_importance_top20.png
output/final_model/interpretability/shap_mean_abs_importance.csv
output/final_model/interpretability/shap_summary_bar.png
output/final_model/interpretability/shap_summary_beeswarm.png
```

Feature importance and SHAP analysis showed that Cp prediction is mainly influenced by descriptors related to:

```text
molecular size
number of atoms/electrons
molecular surface area
molecular shape/connectivity
rotational flexibility
polarity and heteroatom composition
```

This is chemically reasonable because heat capacity generally increases with molecular size and the number of accessible molecular degrees of freedom.

---

## 14. Evaluation Strategy

The model is evaluated in three ways:

### 14.1 GroupKFold OOF CV

The pipeline uses GroupKFold based on canonical SMILES to avoid placing the same compound in both training and validation folds.

This produces:

```text
output/final_model/final_regressor_oof_predictions.csv
output/final_model/parity_oof_cv.png
```

### 14.2 Train/Validation/Test split

The final model is also evaluated using a train/validation/test split.

```text
train      60%
validation 20%
test       20%
```

The split is performed by canonical SMILES groups to reduce compound leakage.

Outputs:

```text
output/final_model/train_validation_test_predictions.csv
output/final_model/parity_train_set.png
output/final_model/parity_validation_set.png
output/final_model/parity_test_set.png
```

### 14.3 Cp-bin error analysis

To check whether the model only performs well in dense Cp regions, the pipeline saves error statistics by Cp quantile bins.

Outputs:

```text
output/final_model/oof_error_by_cp_bin.csv
output/final_model/train_error_by_cp_bin.csv
output/final_model/validation_error_by_cp_bin.csv
output/final_model/test_error_by_cp_bin.csv
```

---

## 15. Important Notes

- The final model does not use phase labels as input.
- Phase classification was not used in the submitted model.
- The model uses SMILES-derived RDKit descriptors.
- Compound names are converted to SMILES before prediction.
- External test prediction uses the saved model and does not retrain the model.
- If external data contain only names, name-to-SMILES resolution is performed before prediction.
- If external data contain SMILES, the pipeline uses the provided SMILES directly.

---

## 16. Clean Submission Recommendation

For final submission, include:

```text
configs/
data/
scripts/
src/
models/final_model/
output/feature_search/
output/final_model/
README.md
run.sh
how_to_run_external.txt
```

Optional development outputs such as the full sweep directories may be excluded to reduce size:

```text
output/final_model_sweep/
output/final_model_sweep_compact/
trial/
__pycache__/
```

The sweep script itself can remain in `scripts/` for reproducibility.

---

## 17. Quick Commands

Run full pipeline:

```bash
bash run.sh
```

Train final model only:

```bash
python scripts/02_train_final_regressor.py \
  --config configs/final_train.yaml
```

Generate final report:

```bash
python scripts/03_make_final_report.py \
  --config configs/final_train.yaml
```

Predict from SMILES:

```bash
python scripts/04_predict_from_smiles.py \
  --config configs/inference.yaml \
  --smiles "CC(=O)C"
```

Predict from compound name:

```bash
python scripts/05_predict_from_name.py \
  --config configs/inference.yaml \
  --name "acetone"
```

Predict external test file:

```bash
python scripts/06_predict_external_test.py \
  --config configs/inference.yaml \
  --input-file data/external/test_all.xlsx \
  --output-dir output/external_test
```

Optional final model sweep:

```bash
python scripts/07_sweep_final_models.py \
  --config configs/final_train.yaml \
  --out-dir output/final_model_sweep_compact
```
