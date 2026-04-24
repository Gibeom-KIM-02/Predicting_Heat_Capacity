import asyncio
import httpx
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse
import re
import time
import os
from thermo.chemical import Chemical
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'mid_project_sample_data.xlsx'
OUTPUT_FILE = 'mid_project_chemeo_filled.xlsx'
STRICT_OUTPUT_FILE = 'mid_project_chemeo_filled_strict.xlsx'
CONCURRENCY_LIMIT = 5
DELAY = 0.5

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_cas_from_thermo(name_or_cas):
    try:
        chem = Chemical(name_or_cas)
        if chem and chem.CAS:
            return chem.CAS
    except Exception:
        pass
    return None

def extract_numeric(text):
    if pd.isna(text): return np.nan
    clean_text = str(text).split('±')[0]
    match = re.search(r"[-+]?\d*\.\d+|\d+", clean_text)
    if match: return float(match.group())
    return np.nan

def process_smiles_rdkit(smiles):
    """Compute molecular descriptors from SMILES using RDKit."""
    if not smiles or pd.isna(smiles):
        return {
            'canonical_smiles': None,
            'molecular_weight': np.nan,
            'rotatable_bonds': np.nan,
            'h_bond_donors': np.nan,
            'h_bond_acceptors': np.nan,
            'tpsa': np.nan,
            'logp': np.nan
        }
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return {
                'canonical_smiles': Chem.MolToSmiles(mol, canonical=True),
                'molecular_weight': round(float(Descriptors.MolWt(mol)), 2),
                'rotatable_bonds': int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
                'h_bond_donors': int(rdMolDescriptors.CalcNumHBD(mol)),
                'h_bond_acceptors': int(rdMolDescriptors.CalcNumHBA(mol)),
                'tpsa': round(float(Descriptors.TPSA(mol)), 2),
                'logp': round(float(Descriptors.MolLogP(mol)), 2)
            }
    except Exception:
        pass
    return {
        'canonical_smiles': None,
        'molecular_weight': np.nan,
        'rotatable_bonds': np.nan,
        'h_bond_donors': np.nan,
        'h_bond_acceptors': np.nan,
        'tpsa': np.nan,
        'logp': np.nan
    }

# ==========================================
# ASYNC SCRAPING LOGIC
# ==========================================
async def fetch_chemeo_data(client, row_data, index, total):
    name = row_data.get('Name', row_data.get('name', 'Unknown'))
    cas_candidate = row_data.get('CAS', row_data.get('cas', name))
    cas = get_cas_from_thermo(cas_candidate)
    
    # Initialize target extraction fields
    extracted = {
        'found_chemeo': False,
        'smiles_chemeo': np.nan,
        'density': np.nan,
        'vapor_pressure': np.nan,
        'melting_point': np.nan,
        'boiling_point': np.nan,
        'critical_temperature': np.nan,
        'critical_pressure': np.nan,
        'enthalpy_of_formation': np.nan,
        'heat_capacity': np.nan,
        'heat_capacity_state': np.nan,
        'viscosity': np.nan,
        'thermal_conductivity': np.nan
    }
    
    if not cas:
        print(f"[{index}/{total}] Failed: Cannot find CAS for {name}")
        return extracted

    search_url = f"https://www.chemeo.com/search?q={urllib.parse.quote(cas)}"
    
    try:
        resp = await client.get(search_url, follow_redirects=True, timeout=20.0)
        if "search" in str(resp.url):
            print(f"[{index}/{total}] Not Found in Chemeo: {cas} ({name})")
            return extracted

        extracted['found_chemeo'] = True

        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')

        for table in tables:
            # Locate table header indexes (dynamic column mapping)
            header_row = table.find('tr')
            if not header_row: continue
            
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            idx_prop = headers.index('property') if 'property' in headers else -1
            idx_val = headers.index('value') if 'value' in headers else -1
            idx_unit = headers.index('unit') if 'unit' in headers else -1
            idx_temp = headers.index('temperature (k)') if 'temperature (k)' in headers else -1
            idx_press = headers.index('pressure (kpa)') if 'pressure (kpa)' in headers else -1
            
            if idx_prop == -1 or idx_val == -1: continue

            # Parse table rows
            for tr in table.find_all('tr')[1:]:
                cols = tr.find_all('td')
                if not cols or len(cols) <= max(idx_prop, idx_val): continue
                
                prop_label = cols[idx_prop].get_text(strip=True).lower()
                val_text = cols[idx_val].get_text(strip=True)
                
                # Extract metadata (unit, temperature, pressure)
                unit_text = cols[idx_unit].get_text(strip=True).lower() if idx_unit != -1 and len(cols) > idx_unit else ''
                temp_val = extract_numeric(cols[idx_temp].get_text(strip=True)) if idx_temp != -1 and len(cols) > idx_temp else np.nan
                press_val = extract_numeric(cols[idx_press].get_text(strip=True)) if idx_press != -1 and len(cols) > idx_press else np.nan
                
                # 1) SMILES
                if 'smiles' in prop_label and pd.isna(extracted['smiles_chemeo']):
                    extracted['smiles_chemeo'] = val_text
                    continue
                
                num_val = extract_numeric(val_text)
                if pd.isna(num_val): continue

                # 2) Temperature properties (melting, boiling, critical) - convert K to C
                if 'melting point' in prop_label or 'tfus' in prop_label:
                    if pd.isna(extracted['melting_point']): 
                        extracted['melting_point'] = round(num_val - 273.15, 2) if 'c' not in unit_text else num_val
                elif 'boiling point' in prop_label or 'tboil' in prop_label:
                    if pd.isna(extracted['boiling_point']): 
                        extracted['boiling_point'] = round(num_val - 273.15, 2) if 'c' not in unit_text else num_val
                elif 'critical temperature' in prop_label or 'tc' in prop_label:
                    if pd.isna(extracted['critical_temperature']):
                        extracted['critical_temperature'] = round(num_val - 273.15, 2) if 'c' not in unit_text else num_val
                
                # 3) Additional properties
                elif 'density' in prop_label:
                    if pd.isna(extracted['density']):
                        extracted['density'] = num_val
                elif 'vapor pressure' in prop_label or 'vapour pressure' in prop_label:
                    if pd.isna(extracted['vapor_pressure']):
                        extracted['vapor_pressure'] = num_val
                elif 'critical pressure' in prop_label or 'pc' in prop_label:
                    if pd.isna(extracted['critical_pressure']):
                        extracted['critical_pressure'] = num_val
                elif 'enthalpy of formation' in prop_label or 'dhf' in prop_label:
                    if pd.isna(extracted['enthalpy_of_formation']):
                        extracted['enthalpy_of_formation'] = num_val
                elif 'heat capacity' in prop_label or 'cp' in prop_label:
                    if pd.isna(extracted['heat_capacity']):
                        extracted['heat_capacity'] = num_val
                        if 'liquid' in prop_label or 'liq' in prop_label:
                            extracted['heat_capacity_state'] = 'liquid'
                        elif 'gas' in prop_label or 'vapor' in prop_label or 'vapour' in prop_label:
                            extracted['heat_capacity_state'] = 'gas'
                        elif 'solid' in prop_label:
                            extracted['heat_capacity_state'] = 'solid'
                        else:
                            extracted['heat_capacity_state'] = 'unknown'
                elif 'viscosity' in prop_label:
                    if pd.isna(extracted['viscosity']):
                        extracted['viscosity'] = num_val
                elif 'thermal conductivity' in prop_label:
                    if pd.isna(extracted['thermal_conductivity']):
                        extracted['thermal_conductivity'] = num_val

        print(f"[{index}/{total}] Scraped successfully: {cas} ({name})")
        return extracted

    except Exception as e:
        print(f"[{index}/{total}] Error during Chemeo request for {cas}: {str(e)[:30]}")
        return extracted

# ==========================================
# MAIN PIPELINE
# ==========================================
async def main():
    print("--- Starting Molecule Strict Extraction Pipeline ---")
    
    try:
        _, input_ext = os.path.splitext(INPUT_FILE.lower())
        if input_ext in ['.xlsx', '.xls']:
            df_raw = pd.read_excel(INPUT_FILE, header=None)
        elif input_ext == '.csv':
            df_raw = pd.read_csv(INPUT_FILE, header=None)
        else:
            raise ValueError(f"Unsupported input file extension: {input_ext}")
        
        # Extract header (row 0) and units (row 1), then use data rows (2+)
        header_row = df_raw.iloc[0]
        units_row_input = df_raw.iloc[1]
        df = df_raw.iloc[2:].reset_index(drop=True)
        df.columns = header_row.values
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total = len(df)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        tasks = []
        
        async def sem_task(row, idx):
            async with semaphore:
                await asyncio.sleep(DELAY)
                res = await fetch_chemeo_data(client, row, idx, total)
                return idx, res

        for idx, row in df.iterrows():
            tasks.append(sem_task(row.to_dict(), idx + 1))
            
        scraped_results = await asyncio.gather(*tasks)

    scraped_results.sort(key=lambda x: x[0])
    
    # Convert scraped data to a DataFrame
    results_df = pd.DataFrame([res[1] for res in scraped_results])
    
    # Compute RDKit descriptors from SMILES
    print("\n--- Applying RDKit Calculations ---")
    # Prefer Chemeo SMILES; fallback to input SMILES if Chemeo SMILES is missing.
    smiles_col = next((c for c in df.columns if c.lower() == 'smiles'), None)
    rdkit_results = []
    for idx, row in df.iterrows():
        smiles_chemeo = results_df.iloc[idx]['smiles_chemeo']
        input_smiles = row[smiles_col] if smiles_col and smiles_col in row.index else np.nan
        smiles_for_rdkit = smiles_chemeo if pd.notna(smiles_chemeo) else input_smiles
        rdkit_results.append(process_smiles_rdkit(smiles_for_rdkit))
    
    # Define output schema with units
    output_columns = [
        'Name', 'SMILES', 'Canonical_SMILES', 'Formula', 'Molecular_weight', 
        'Rotatable_bonds', 'H_bond_donors', 'H_bond_acceptors', 'TPSA', 'LogP',
        'Melting_point', 'Boiling_point', 'Density', 'Vapor_pressure', 
        'Enthalpy_of_formation', 'Heat_capacity', 'Heat_capacity_state', 'Critical_temperature', 
        'Critical_pressure', 'Acentric_factor', 'Viscosity', 'Thermal_conductivity', 'Price'
    ]
    
    units_data = {
        'Name': '', 'SMILES': '', 'Canonical_SMILES': '', 'Formula': '', 
        'Molecular_weight': 'g/mol', 'Rotatable_bonds': '', 'H_bond_donors': '', 
        'H_bond_acceptors': '', 'TPSA': 'Ų', 'LogP': '',
        'Melting_point': '°C', 'Boiling_point': '°C', 'Density': 'g/cm³', 
        'Vapor_pressure': 'Pa', 'Enthalpy_of_formation': 'kJ/mol', 
        'Heat_capacity': 'J/(mol·K)', 'Heat_capacity_state': '', 'Critical_temperature': '°C', 
        'Critical_pressure': 'Pa', 'Acentric_factor': '', 'Viscosity': 'Pa·s', 
        'Thermal_conductivity': 'W/(m·K)', 'Price': ''
    }
    
    # Merge with input file units
    for col in output_columns:
        if col in units_row_input.index and pd.notna(units_row_input[col]):
            units_data[col] = units_row_input[col]
    
    # Identify input column names (case-insensitive)
    name_col = next((c for c in df.columns if c.lower() == 'name'), None)
    formula_col = next((c for c in df.columns if c.lower() == 'formula'), None)
    price_col = next((c for c in df.columns if c.lower() == 'price'), None)
    acentric_col = next((c for c in df.columns if c.lower() == 'acentric_factor'), None)
    
    def get_value(row, col, default=np.nan):
        return row[col] if col and col in row.index else default
    
    # Build non-strict and strict outputs
    data_rows_non_strict = []
    data_rows_strict = []
    
    for idx, row in df.iterrows():
        scraped = results_df.iloc[idx]
        rdkit = rdkit_results[idx]
        found_chemeo = scraped['found_chemeo']
        
        # Build row data with priority: Chemeo > RDKit > Input
        row_data = {
            'Name': get_value(row, name_col),
            'SMILES': get_value(row, smiles_col),
            'Canonical_SMILES': rdkit['canonical_smiles'] if rdkit['canonical_smiles'] else get_value(row, smiles_col),
            'Formula': get_value(row, formula_col),
            'Molecular_weight': rdkit['molecular_weight'] if not pd.isna(rdkit['molecular_weight']) else np.nan,
            'Rotatable_bonds': rdkit['rotatable_bonds'] if not pd.isna(rdkit['rotatable_bonds']) else np.nan,
            'H_bond_donors': rdkit['h_bond_donors'] if not pd.isna(rdkit['h_bond_donors']) else np.nan,
            'H_bond_acceptors': rdkit['h_bond_acceptors'] if not pd.isna(rdkit['h_bond_acceptors']) else np.nan,
            'TPSA': rdkit['tpsa'] if not pd.isna(rdkit['tpsa']) else np.nan,
            'LogP': rdkit['logp'] if not pd.isna(rdkit['logp']) else np.nan,
            'Melting_point': scraped['melting_point'] if not pd.isna(scraped['melting_point']) else np.nan,
            'Boiling_point': scraped['boiling_point'] if not pd.isna(scraped['boiling_point']) else np.nan,
            'Density': scraped['density'] if not pd.isna(scraped['density']) else np.nan,
            'Vapor_pressure': scraped['vapor_pressure'] if not pd.isna(scraped['vapor_pressure']) else np.nan,
            'Enthalpy_of_formation': scraped['enthalpy_of_formation'] if not pd.isna(scraped['enthalpy_of_formation']) else np.nan,
            'Heat_capacity': scraped['heat_capacity'] if not pd.isna(scraped['heat_capacity']) else np.nan,
            'Heat_capacity_state': scraped['heat_capacity_state'] if pd.notna(scraped['heat_capacity_state']) else np.nan,
            'Critical_temperature': scraped['critical_temperature'] if not pd.isna(scraped['critical_temperature']) else np.nan,
            'Critical_pressure': scraped['critical_pressure'] if not pd.isna(scraped['critical_pressure']) else np.nan,
            'Acentric_factor': get_value(row, acentric_col),
            'Viscosity': scraped['viscosity'] if not pd.isna(scraped['viscosity']) else np.nan,
            'Thermal_conductivity': scraped['thermal_conductivity'] if not pd.isna(scraped['thermal_conductivity']) else np.nan,
            'Price': get_value(row, price_col)
        }
        
        data_rows_non_strict.append(row_data)
        
        # For strict: replace Chemeo fields with '-' if not found
        if not found_chemeo:
            row_data_strict = row_data.copy()
            row_data_strict.update({
                'Melting_point': '-', 'Boiling_point': '-', 'Density': '-',
                'Vapor_pressure': '-', 'Enthalpy_of_formation': '-', 'Heat_capacity': '-',
                'Heat_capacity_state': '-',
                'Critical_temperature': '-', 'Critical_pressure': '-', 'Viscosity': '-',
                'Thermal_conductivity': '-'
            })
            data_rows_strict.append(row_data_strict)
        else:
            data_rows_strict.append(row_data)
    
    # Build DataFrames
    df_non_strict = pd.DataFrame(data_rows_non_strict)[output_columns]
    df_strict = pd.DataFrame(data_rows_strict)[output_columns]

    # Strict filter: drop rows missing MW, rotatable bonds, mp, bp, Tc, or Cp.
    required_strict_cols = [
        'Molecular_weight', 'Rotatable_bonds', 'Melting_point',
        'Boiling_point', 'Critical_temperature', 'Heat_capacity'
    ]
    strict_total_before_filter = len(df_strict)
    strict_missing_mask = pd.Series(False, index=df_strict.index)
    for col in required_strict_cols:
        col_series = df_strict[col]
        strict_missing_mask = strict_missing_mask | col_series.isna() | (col_series.astype(str).str.strip() == '-') | (col_series.astype(str).str.strip() == '')
    strict_removed_count = int(strict_missing_mask.sum())
    df_strict = df_strict[~strict_missing_mask].reset_index(drop=True)
    strict_remaining_count = len(df_strict)
    print(
        f"Strict filter summary: total={strict_total_before_filter}, "
        f"removed={strict_removed_count}, remaining={strict_remaining_count}"
    )
    
    # Build output with header row and units row only
    header_row_dict = {col: col for col in output_columns}
    units_row_dict = {col: units_data[col] for col in output_columns}
    
    # Prepend header and units rows
    df_non_strict = pd.concat([
        pd.DataFrame([header_row_dict]),
        pd.DataFrame([units_row_dict]),
        df_non_strict
    ], ignore_index=True)
    
    df_strict = pd.concat([
        pd.DataFrame([header_row_dict]),
        pd.DataFrame([units_row_dict]),
        df_strict
    ], ignore_index=True)

    def save_by_extension(dataframe, path):
        _, ext = os.path.splitext(path.lower())
        if ext in ['.xlsx', '.xls']:
            dataframe.to_excel(path, index=False, header=False)
        elif ext == '.csv':
            dataframe.to_csv(path, index=False, header=False)
        else:
            raise ValueError(f"Unsupported output file extension: {ext}")

    save_by_extension(df_non_strict, OUTPUT_FILE)
    save_by_extension(df_strict, STRICT_OUTPUT_FILE)

    print(f"\n✅ Extraction Complete! Saved non-strict output: '{OUTPUT_FILE}'.")
    print(f"✅ Strict output saved: '{STRICT_OUTPUT_FILE}'.")

if __name__ == "__main__":
    start_time = time.time()
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    print(f"Total time: {time.time() - start_time:.2f} seconds")