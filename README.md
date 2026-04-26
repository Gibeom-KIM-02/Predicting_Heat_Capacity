# AIML Midterm Project: Chemical Heat Capacity Prediction

This repository contains the workflow for building a machine learning regression model to predict the constant-pressure heat capacity, **Cp**, of chemical compounds in **J/mol·K**.

The project includes dataset polishing, feature evaluation, data extraction from external sources, final data cleaning, model training, and prediction code.

## Directory Structure

```text
3.midterm_final/
├─ 1.polish_initial_dataset/
├─ 2.feature_evaluation_for_initial_data/
├─ 3_1.data_extraction_from_NIST/
├─ 3_2.data_extraction_from_chemco/
├─ 3_3.data_extraction_for_gas_from_DFT_data/
├─ 3_4.data_polish/
├─ 4_1.Cp_predicting_code/
├─ 4_2.Cp_predicting_model/
├─ requirements.txt
├─ environment.yml
└─ README.md
```

### Directory Description

| Directory | Description |
|---|---|
| `1.polish_initial_dataset/` | Cleaning and polishing the initial dataset |
| `2.feature_evaluation_for_initial_data/` | Evaluating candidate physical features and descriptor/model combinations |
| `3_1.data_extraction_from_NIST/` | Extracting heat capacity data from NIST/ThermoML sources |
| `3_2.data_extraction_from_chemco/` | Extracting additional heat capacity data from Chemeo |
| `3_3.data_extraction_for_gas_from_DFT_data/` | Preparing ideal-gas heat capacity data from DFT-based sources |
| `3_4.data_polish/` | Merging, deduplicating, and polishing the final dataset |
| `4_1.Cp_predicting_code/` | Training and evaluation code for the final Cp prediction model |
| `4_2.Cp_predicting_model/` | Saved final model, inference configuration, and prediction workflow |

---

## Environment Setup

This project was designed to run in a conda environment with Python 3.10 or 3.11.  
Python 3.11 is recommended.

### Recommended installation

The recommended method is to use `environment.yml`, because RDKit is more stable when installed through conda-forge.

```bash
conda env create -f environment.yml
conda activate AIML_midterm
```

### Alternative installation using requirements.txt

If `environment.yml` is not used, install RDKit first through conda-forge and then install the remaining Python packages from `requirements.txt`.

```bash
conda create -n AIML_midterm python=3.11 -y
conda activate AIML_midterm

conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

## Required Packages

The main Python dependencies are:

```text
numpy
pandas
openpyxl
PyYAML
joblib
scikit-learn
matplotlib
shap
rdkit
requests
httpx
beautifulsoup4
pubchempy
```

### Package roles

| Package | Role |
|---|---|
| `numpy`, `pandas` | Numerical calculation and dataframe handling |
| `openpyxl` | Reading and writing Excel files |
| `PyYAML` | Reading YAML configuration files |
| `joblib` | Saving and loading trained models |
| `scikit-learn` | Machine learning models and evaluation |
| `matplotlib` | Plot generation |
| `shap` | SHAP feature-importance analysis |
| `rdkit` | SMILES parsing and molecular descriptor calculation |
| `requests`, `httpx` | Web/API requests |
| `beautifulsoup4` | HTML parsing |
| `pubchempy` | Compound name to PubChem information/SMILES lookup |

## Notes on RDKit

RDKit can sometimes be difficult to install with only pip, especially on Linux or HPC systems. Therefore, the preferred installation method is:

```bash
conda install -c conda-forge rdkit
```

This avoids most compatibility issues related to compiled chemistry libraries.

## General Workflow

The project is organized into several stages:

1. **Initial dataset polishing**  
   Clean and prepare the provided initial dataset.

2. **Feature evaluation**  
   Test candidate physical features and descriptor-based feature sets using regression models.

3. **Data extraction**  
   Collect or process additional Cp data from NIST, ChemEO, and DFT-based gas-phase data.

4. **Data polishing**  
   Merge, clean, deduplicate, and organize the final dataset.

5. **Cp prediction code**  
   Train and evaluate machine learning models for Cp prediction.

6. **Final prediction model**  
   Use the trained model to predict Cp from compound name or SMILES.

---

## Example Usage

Activate the environment first:

```bash
conda activate AIML_midterm
```

Then move into the directory containing the target script and run it according to the instructions in that subdirectory.

Example:

```bash
cd 4_1.Cp_predicting_code
python scripts/01_run_feature_search.py --config configs/feature_search.yaml
```

For prediction using the final model, use the scripts and configuration files in:

```text
4_2.Cp_predicting_model/
```

---

## Reproducibility

All major Python dependencies are summarized in:

```text
requirements.txt
```

For the most reproducible setup, use:

```text
environment.yml
```

The conda-based setup is recommended because this project depends on RDKit for molecular descriptor generation.
