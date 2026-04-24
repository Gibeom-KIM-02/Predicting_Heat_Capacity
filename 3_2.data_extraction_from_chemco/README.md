# Chemeo scraping refactor

## Directory layout

- `input/` : input CSV files
- `output/` : main outputs and split CSVs
- `logs/` : scraper logs
- `scripts/` : executable entry-point scripts
- `src/` : reusable modules

## Expected workflow

1. Put `PubChem_compound_CID.csv` inside `input/`
2. Run the scraper:
   ```bash
   python scripts/01_fetch_chemeo_async.py
   ```
3. Split Cp by phase:
   ```bash
   python scripts/02_split_cp_by_phase.py
   ```
4. Or, You can easily run it with
   ```bash
   bash ./run.sh
   ```

## Why this layout is cleaner

- Paths are centralized in `config.yaml`
- Scraper logic is separated from file I/O utilities
- Parsing logic is separated from request / orchestration logic
- Post-processing is an independent script
