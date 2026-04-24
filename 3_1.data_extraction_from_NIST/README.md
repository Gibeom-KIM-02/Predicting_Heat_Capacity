# Refactored Cp Master Pipeline

## Structure

```bash
cp_pipeline_refactored_v2/
├─ Build_Cp_Master_Pipeline.py
├─ build_cp.example.yaml
├─ input/
│  ├─ 10.1007/
│  ├─ 10.1016/
│  └─ 10.1021/
├─ output/
│  ├─ raw_master/
│  ├─ clean_master/
│  └─ derived_datasets/
└─ scripts/
   ├─ config_utils.py
   ├─ structure_features.py
   ├─ thermoml_parser.py
   ├─ cleaning.py
   ├─ dataset_builder.py
   ├─ summary_writer.py
   └─ utils.py        
```

- `Build_Cp_Master_Pipeline.py`: thin entrypoint and top-level orchestration of the full pipeline
- `scripts/config_utils.py`: configuration loading, path resolution, and input/output directory preparation
- `scripts/utils.py`: shared utility helpers and common constants such as normalization rules, safe converters, and output column definitions
- `scripts/structure_features.py`: RDKit structure parsing and molecular descriptor calculation
- `scripts/thermoml_parser.py`: ThermoML JSON parsing and raw master dataset construction
- `scripts/cleaning.py`: clean-master filtering, validation, and exact-deduplication logic
- `scripts/dataset_builder.py`: temperature-window and representative dataset generation, including outlier filtering and selection rules
- `scripts/summary_writer.py`: summary CSV generation for numeric, phase, compound, and overall dataset statistics

## Output Directories

The pipeline writes all generated results into the `output/` directory.  
Each subdirectory represents a different stage of data processing.

---

### `raw_master/`

Contains the raw Cp dataset extracted directly from ThermoML JSON files **before any filtering**.

#### Typical contents
- `cp_master_all_phases.csv`
- `cp_master_all_phases.xlsx`
- `cp_master_file_stats.csv`
- numeric and phase summary CSV files

#### Purpose
- Preserve the original parsed data
- Allow inspection of all extracted Cp-related rows
- Serve as a debugging reference if later filtering removes data

---

### `clean_master/`

Contains the cleaned dataset after applying validation and filtering rules.

#### Typical contents
- `cp_master_clean.csv`
- `cp_master_clean.xlsx`
- `cp_master_clean_filter_log.csv`
- numeric, phase, compound, and overall summary CSV files

#### Purpose
- Remove invalid or incomplete rows
- Enforce constraints such as:
  - valid temperature range
  - valid pressure range
  - valid Cp range
  - allowed phase types
- Provide a high-quality dataset for downstream analysis

---

### `derived_datasets/`

Contains datasets derived from the cleaned master dataset.

#### Typical contents
- `cp_master_298K_window.csv`
- `cp_master_298K_window.xlsx`
- `cp_master_ambient_representative.csv`
- `cp_master_ambient_representative.xlsx`
- dataset-specific filter logs and summaries

#### Purpose
- Generate analysis-ready subsets
- Select data near a reference temperature (e.g., 298.15 K)
- Construct representative datasets (e.g., one row per compound or compound-phase group)

---

### Summary

- `raw_master` → direct ThermoML parsing results  
- `clean_master` → validated and filtered dataset  
- `derived_datasets` → analysis-ready subsets  

This separation ensures:
- reproducibility
- traceability
- easier debugging and validation

---

## Run

```bash
python Build_Cp_Master_Pipeline.py --config build_cp.yaml
```
