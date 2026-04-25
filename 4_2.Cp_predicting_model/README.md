# Cp Prediction Runtime Package

This directory provides a packaged runtime version of the Cp prediction model.

The model predicts constant-pressure heat capacity, Cp, from compound information such as SMILES or compound names. The prediction code and Python environment are packaged inside the Apptainer/Singularity image:

```bash
cp_predictor_runtime.sif
```

Therefore, users do not need to install Python packages manually if Apptainer/Singularity is available on the system.

---

## 1. Directory Structure

```text
4_2.Cp_predicting_model/
├── configs/
│   └── inference.yaml
├── cp_predictor_runtime.sif
├── data/
│   ├── external/
│   │   └── test_all.xlsx
│   └── processed/
│       └── Cp_input_data_final.xlsx
├── output/
│   └── external_test_sif/
├── run_sif_external.sh
└── README.md
```

### Main files

| File or directory | Description |
|---|---|
| `cp_predictor_runtime.sif` | Packaged Apptainer/Singularity image containing the prediction code, model, and required Python libraries |
| `configs/inference.yaml` | Configuration file for inference |
| `data/external/test_all.xlsx` | Input Excel file for external prediction |
| `data/processed/Cp_input_data_final.xlsx` | Local lookup table used for compound name to SMILES conversion |
| `output/external_test_sif/` | Output directory for prediction results |
| `run_sif_external.sh` | Shell script for running the packaged prediction model |

---

## 2. Input File

The input file should be placed at:

```bash
data/external/test_all.xlsx
```

The input Excel file should contain at least one of the following SMILES columns:

```text
canonical_smiles
smiles
SMILES
```

If SMILES information is not available, the program can try to use compound name columns such as:

```text
chemical_compound_name
compound_name
name
```

If the input file contains true Cp values, the program will also calculate prediction errors and generate a parity plot.
Recognized target column names are:

```text
Cp [J/mol*K]
Cp_J_molK
heat_capacity
Cp
```

---

## 3. Configuration File

The inference configuration is located at:

```bash
configs/inference.yaml
```

Current configuration:

```yaml
model_dir: models/final_model
output_dir: output/prediction_examples
feature_set: selected_rdkit_v1

smiles_columns:
  - canonical_smiles
  - smiles
  - SMILES

name_columns:
  - chemical_compound_name
  - compound_name
  - name

target_columns:
  - Cp [J/mol*K]
  - Cp_J_molK
  - heat_capacity
  - Cp

name_resolution:
  use_local_lookup: true
  local_lookup_files:
    - data/processed/Cp_input_data_final.xlsx
  sheet_name: null
  allow_pubchem_fallback: true
```

The model itself is already included inside the `.sif` image.
The `data/processed/Cp_input_data_final.xlsx` file is used only when compound names need to be resolved into SMILES.

---

## 4. How to Run

From the project root directory, run:

```bash
bash run_sif_external.sh
```

The script runs:

```bash
apptainer run \
  --bind "${PROJECT_ROOT}/output:/app/output" \
  --bind "${PROJECT_ROOT}/data:/app/data" \
  cp_predictor_runtime.sif \
  --config configs/inference.yaml \
  --input-file data/external/test_all.xlsx \
  --output-dir output/external_test_sif
```

The `--bind` options are important because the `.sif` image is read-only. Input and output folders must be connected to the container through bind mounts.

---

## 5. Output Files

After running the script, results are saved in:

```bash
output/external_test_sif/
```

Expected output files:

```text
external_test_predictions.csv
external_test_summary.csv
parity_external_test.png
```

### Output description

| Output file | Description |
|---|---|
| `external_test_predictions.csv` | Predicted Cp values for each input compound |
| `external_test_summary.csv` | Summary of prediction performance, if true Cp values are available |
| `parity_external_test.png` | Parity plot comparing true Cp and predicted Cp, if true Cp values are available |

---

## 6. Notes

This package is intended for inference only.

It does not perform feature search, model training, or model selection.

The training pipeline, feature search results, and model development files are not required to run this package. Only the input Excel file, configuration file, local lookup data, output directory, and `.sif` image are needed.

---

## 7. Quick Test

To check whether the package works correctly, run:

```bash
bash run_sif_external.sh
```

If the run is successful, the following files should appear:

```bash
ls output/external_test_sif
```

Expected result:

```text
external_test_predictions.csv
external_test_summary.csv
parity_external_test.png
```

