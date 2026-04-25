# Gas-phase Cp298 Dataset Processing Pipeline

This project processes a computational thermodynamic dataset to extract heat capacity at 298 K (Cp298) and augment it with canonical SMILES using PubChem + RDKit.

---

## 1. Data Source

The input dataset:

- `ComputedFormationEnthalpies_2022Feb.csv`

Source:
- https://catalog.data.gov/dataset/ideal-gas-thermodynamic-properties-for-organic-compounds-containing-up-to-7-c-o-or-n-atoms

Description:
- Ideal-gas thermodynamic properties
- Computed using ab initio quantum chemistry methods
- Contains:
  - Enthalpy (H)
  - Entropy (S)
  - Heat capacity (Cp at 298.15 K)

Important:
- The file contains multiple calculation method blocks marked by:

```
   MODEL: ...
```

- The same molecule may appear multiple times across different methods.

---

## 2. Project Structure

```
4.gas_compute_chem/  
├── 01_organise_data_with_smiles.py  
├── input/  
│   └── ComputedFormationEnthalpies_2022Feb.csv  
├── output/  
│   ├── Cp298_with_SMILES_rows.csv  
│   ├── Cp298_with_SMILES_unique_by_method.csv  
│   ├── Cp298_with_SMILES_unique_all_methods_mean.csv  
│   └── Cp298_with_SMILES.xlsx  
├── note.txt  
└── README.md  
```

---

## 3. What the Script Does

Step 1 — Parse CSV with method blocks  
- Detects `# MODEL:` lines  
- Assigns `calculation_method` and `method_block_id`  

Step 2 — Clean and extract data  
- formula  
- chemical_compound_name  
- Cp_298_J_per_molK  

Step 3 — Query PubChem  
- Retrieves SMILES, CID, IUPAC name  

Step 4 — Canonicalize SMILES (RDKit)  
- Uses canonical SMILES for consistency  

Step 5 — Generate Outputs  

(1) Cp298_with_SMILES_rows.csv  
- All rows, includes method info  

(2) Cp298_with_SMILES_unique_by_method.csv  
- Grouped by method  

(3) Cp298_with_SMILES_unique_all_methods_mean.csv  
- Averaged across methods  

(4) Cp298_with_SMILES.xlsx  
- Combined Excel output  

---

## 4. How to Run

python 01_organise_data_with_smiles.py

Requirements:
- Python 3.9+
- pandas
- pubchempy
- rdkit

---

## 5. Design Notes

- Method blocks are preserved (no mixing across methods)
- Canonical SMILES ensures consistent molecular representation
- PubChem queries are cached
