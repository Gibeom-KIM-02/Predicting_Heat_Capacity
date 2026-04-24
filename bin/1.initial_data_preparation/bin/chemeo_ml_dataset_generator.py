import asyncio
import httpx
import pandas as pd
from bs4 import BeautifulSoup
import urllib.parse
import re
import time
import random
from chemicals.critical import critical_data_Yaws
try:
    from thermo.chemical import Chemical
except Exception:
    Chemical = None
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:
    Chem = None
    Descriptors = None
    rdMolDescriptors = None

# ==========================================
# CONFIGURATION
# ==========================================
# Process all available CAS numbers from Yaws (7549 total).
# Collect all compounds that meet criteria: SMILES + at least one Cp value
CONCURRENCY_LIMIT = 10     # Number of simultaneous web requests
OUTPUT_FILE = 'chemeo_raw_extracted.xlsx'
ATM_PRESSURE_KPA = 101.325
PRESSURE_TOLERANCE_KPA = 10.0

OUTPUT_COLUMNS = [
    'name', 'CAS', 'SMILES', 'melting point', 'boiling point',
    'Cp_solid', 'Cp_liquid', 'Cp_gas', 'phase_at_25°C'
]

STEP2_OUTPUT_COLUMNS = [
    'name', 'CAS', 'SMILES', 'canonical_smiles', 'MW', 'rotatable_bonds',
    'melting point', 'boiling point', 'Cp_solid', 'Cp_liquid', 'Cp_gas', 'phase_at_25°C'
]

def extract_numeric(text):
    """Extracts the first float/int from a string."""
    if not text:
        return None
    # Remove the uncertainty part (e.g., "± 0.5") before searching
    clean_text = text.split('±')[0]
    match = re.search(r"[-+]?\d*\.\d+|\d+", clean_text)
    if match:
        return float(match.group())
    return None

def extract_numeric_range(text):
    """Extract [low, high] numeric range from text; single values map to (v, v)."""
    if not text:
        return None, None

    range_match = re.search(r"\[\s*([-+]?\d*\.?\d+)\s*;\s*([-+]?\d*\.?\d+)\s*\]", str(text))
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if low > high:
            low, high = high, low
        return low, high

    v = extract_numeric(text)
    if v is None:
        return None, None
    return v, v

def is_pressure_near_atm(pressure_text, pressure_unit_text=''):
    """Check whether the pressure condition is near 1 atm."""
    low, high = extract_numeric_range(pressure_text)
    if low is None or high is None:
        return False

    unit_src = f"{pressure_text} {pressure_unit_text}".lower()
    if 'mpa' in unit_src:
        low_kpa, high_kpa = low * 1000.0, high * 1000.0
    elif 'bar' in unit_src:
        low_kpa, high_kpa = low * 100.0, high * 100.0
    elif 'atm' in unit_src:
        low_kpa, high_kpa = low * 101.325, high * 101.325
    elif 'mmhg' in unit_src or 'torr' in unit_src:
        low_kpa, high_kpa = low * 0.133322, high * 0.133322
    else:
        # In Chemeo pressure-dependent rows for Tboil/Tfus, pressure is typically kPa.
        low_kpa, high_kpa = low, high

    return (low_kpa - PRESSURE_TOLERANCE_KPA) <= ATM_PRESSURE_KPA <= (high_kpa + PRESSURE_TOLERANCE_KPA)

def is_near_atm_row(cols, prop_label):
    """Use row pressure columns when present; otherwise accept plain MP/BP rows."""
    has_pressure_col = len(cols) >= 4 and bool(cols[3].get_text(strip=True))
    if not has_pressure_col:
        # Plain melting/boiling point rows usually represent standard pressure values.
        return True

    pressure_text = cols[3].get_text(strip=True)
    pressure_unit_text = cols[4].get_text(strip=True) if len(cols) >= 5 else ''
    return is_pressure_near_atm(pressure_text, pressure_unit_text)

def is_cp_near_room_temp(cols):
    """Check if Cp temperature is near 298.15K. If not specified, assume room temp."""
    if len(cols) >= 4:
        temp_text = cols[3].get_text(strip=True)
        temp_val = extract_numeric(temp_text)
        if temp_val is not None:
            # 298.15K +- 5K
            if abs(temp_val - 298.15) > 5.0:
                return False
    return True

def get_rdkit_features(smiles):
    """Calculate RDKit descriptors and canonical-SMILES check results."""
    if not smiles or Chem is None or Descriptors is None or rdMolDescriptors is None:
        return None, None, None

    try:
        input_smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            return None, None, None

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        mw = round(float(Descriptors.MolWt(mol)), 2)
        rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        return mw, rot_bonds, canonical_smiles
    except Exception:
        return None, None, None

def normalize_compound_name(name):
    """Normalize compound name formatting for consistent display/output."""
    if name is None:
        return None

    normalized = re.sub(r"\s+", " ", str(name)).strip().strip('.')
    if not normalized:
        return None

    # If thermo returns all-lowercase/all-uppercase names, normalize to sentence case.
    if normalized == normalized.lower() or normalized == normalized.upper():
        normalized = normalized.lower()
        normalized = normalized[0].upper() + normalized[1:]

    return normalized

def normalize_unit_text(unit_text):
    if not unit_text:
        return ''
    return unit_text.lower().replace(' ', '').replace('×', '*')

def convert_temperature_to_c(value, unit_text):
    """Convert temperature to Celsius when possible."""
    if value is None:
        return None

    u = normalize_unit_text(unit_text)
    if not u:
        # Chemeo temperature-dependent sections are usually in K.
        return round(value - 273.15, 2)

    if 'k' in u and 'kg' not in u:
        return round(value - 273.15, 2)
    if 'degc' in u or '°c' in u or (u == 'c'):
        return round(value, 2)
    if 'degf' in u or '°f' in u or (u == 'f'):
        return round((value - 32.0) * 5.0 / 9.0, 2)
    return None

def convert_cp_to_j_mol_k(value, unit_text):
    """Convert Cp to J/mol K when unit is recognizable."""
    if value is None:
        return None

    u = normalize_unit_text(unit_text)
    if not u:
        # Most Chemeo Cp values are already J/mol K.
        return value

    if 'j/mol' in u and 'k' in u and 'kj/mol' not in u:
        return value
    if 'kj/mol' in u and 'k' in u:
        return value * 1000.0
    if 'cal/mol' in u and 'k' in u and 'kcal/mol' not in u:
        return value * 4.184
    if 'kcal/mol' in u and 'k' in u:
        return value * 4184.0

    # Unknown unit type (e.g., mass based units) -> skip instead of storing wrong values.
    return None

def determine_phase_25c(mp_c, bp_c):
    """Infer phase at 25C from melting/boiling points."""
    current_temp = 25.0
    if mp_c is not None and current_temp < mp_c:
        return 'solid'
    if bp_c is not None and current_temp > bp_c:
        return 'gas'
    return 'liquid'

def extract_smiles(soup):
    """Extract the profile SMILES string from the page text."""
    smiles_link = soup.find('a', href=re.compile(r'similar\?smiles='))
    if smiles_link:
        smiles_text = smiles_link.get_text(strip=True)
        if smiles_text and len(smiles_text) <= 200:
            return smiles_text

    page_text = soup.get_text(" ", strip=True)
    match = re.search(r"SMILES\s+([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)", page_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_compound_name(soup):
    """Extract compound name from Cheméo page heading/title."""
    h1 = soup.find('h1')
    if h1:
        h1_text = h1.get_text(' ', strip=True)
        m = re.search(r"chemical\s+properties\s+of\s+(.+?)\s*\(\s*cas", h1_text, re.IGNORECASE)
        if m:
            return normalize_compound_name(m.group(1))
        if h1_text and len(h1_text) < 200:
            return normalize_compound_name(h1_text)

    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text(' ', strip=True)
        m = re.search(r"chemical\s+properties\s+of\s+(.+?)\s*\(\s*cas", title_text, re.IGNORECASE)
        if m:
            return normalize_compound_name(m.group(1))

    return None

def extract_name_from_thermo(cas):
    """Try to get compound name from thermo first."""
    if Chemical is None:
        return None

    try:
        chem = Chemical(cas, T=298.15, P=101325)
        name = getattr(chem, 'name', None)
        if name and str(name).strip():
            return normalize_compound_name(name)
    except Exception:
        return None
    return None

async def fetch_chemeo_properties(client, cas, index, total):
    """Asynchronously fetches SMILES, MP, BP, and Cp from Chemeo using CAS."""
    search_url = f"https://www.chemeo.com/search?q={urllib.parse.quote(cas)}"
    
    extracted = {
        'name': None,
        'CAS': cas,
        'SMILES': None,
        'melting point': None,
        'boiling point': None,
        'Cp_solid': None,
        'Cp_liquid': None,
        'Cp_gas': None,
        'phase_at_25°C': None
    }
    
    try:
        # 0. Try thermo name first; fallback to Cheméo name later.
        extracted['name'] = extract_name_from_thermo(cas)

        # 1. Search and redirect to compound page
        resp = await client.get(search_url, follow_redirects=True, timeout=20.0)
        
        # If the search page didn't redirect to a specific CID, skip
        if "search" in str(resp.url):
            print(f"[{index}/{total}] [Unknown] {cas} - Not found in Chemeo")
            return extracted

        soup = BeautifulSoup(resp.text, 'html.parser')
        if not extracted['name']:
            extracted['name'] = extract_compound_name(soup)
        extracted['SMILES'] = extract_smiles(soup)
        rows = soup.find_all('tr')
        
        # 2. Parse all table rows
        for tr in rows:
            cols = tr.find_all(['th', 'td'])
            if len(cols) < 2:
                continue
                
            prop_label = cols[0].get_text(strip=True).lower()
            raw_val_text = cols[1].get_text(strip=True)
            unit_text = cols[2].get_text(strip=True) if len(cols) >= 3 else ''
            
            # Extract SMILES
            if 'smiles' in prop_label and not extracted['SMILES']:
                extracted['SMILES'] = raw_val_text
                continue
                
            # Extract numeric properties
            val = extract_numeric(raw_val_text)
            if val is None:
                continue

            # Melting Point (Tfus) - Assume K and convert to C
            if 'melting point' in prop_label or 'tfus' in prop_label:
                if extracted['melting point'] is None:
                    if not is_near_atm_row(cols, prop_label):
                        continue
                    temp_c = convert_temperature_to_c(val, unit_text)
                    if temp_c is not None:
                        extracted['melting point'] = temp_c

            # Boiling Point (Tboil) - Assume K and convert to C
            elif 'boiling point' in prop_label or 'tboil' in prop_label:
                if extracted['boiling point'] is None:
                    if not is_near_atm_row(cols, prop_label):
                        continue
                    temp_c = convert_temperature_to_c(val, unit_text)
                    if temp_c is not None:
                        extracted['boiling point'] = temp_c

            # Heat Capacity (Cp)
            elif 'heat capacity' in prop_label or 'cp' in prop_label:
                cp_val = convert_cp_to_j_mol_k(val, unit_text)
                if cp_val is None:
                    continue

                if 'solid' in prop_label or 'cps' in prop_label:
                    if extracted['Cp_solid'] is None:
                        extracted['Cp_solid'] = round(cp_val, 4)
                elif 'liquid' in prop_label or 'cpl' in prop_label:
                    if extracted['Cp_liquid'] is None:
                        extracted['Cp_liquid'] = round(cp_val, 4)
                elif 'gas' in prop_label or 'cpg' in prop_label:
                    if extracted['Cp_gas'] is None:
                        extracted['Cp_gas'] = round(cp_val, 4)

        extracted['phase_at_25°C'] = determine_phase_25c(extracted['melting point'], extracted['boiling point'])

        # Check if we got at least SMILES and one numeric property to consider it a "success"
        has_data = extracted['SMILES'] and (extracted['melting point'] or extracted['boiling point'] or extracted['Cp_liquid'] or extracted['Cp_solid'] or extracted['Cp_gas'])
        name_for_log = extracted['name'] if extracted['name'] else 'Unknown'
        
        if has_data:
            print(f"[{index}/{total}] [{name_for_log}] {cas} - Success")
        else:
            print(f"[{index}/{total}] [{name_for_log}] {cas} - Parsed, but lacking target features")
            
        return extracted

    except Exception as e:
        print(f"[{index}/{total}] [Unknown] {cas} - Request Failed")
        return extracted

async def main():
    print("Starting Asynchronous Chemeo Data Harvester...")
    
    # 1. Get all available CAS numbers from Yaws database
    all_cas_list = critical_data_Yaws.index.dropna().tolist()
    target_cas_list = all_cas_list

    total = len(target_cas_list)
    print(f"Processing all {total} CAS candidates from Yaws. Collecting all compounds that meet criteria (SMILES + at least one Cp value).")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    limits = httpx.Limits(max_keepalive_connections=CONCURRENCY_LIMIT, max_connections=CONCURRENCY_LIMIT*2)
    
    dataset = []
    valid_count = 0

    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        tasks = []
        
        async def sem_task(cas_num, idx):
            async with semaphore:
                # Slight random delay to mimic human requests and avoid IP bans
                await asyncio.sleep(random.uniform(0.1, 0.5))
                return await fetch_chemeo_properties(client, cas_num, idx, total)

        # Build task list
        for i, cas in enumerate(target_cas_list, start=1):
            tasks.append(sem_task(cas, i))
            
        # Execute concurrently
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            
            # Filter logic: Only keep rows that have SMILES and at least ONE Cp value
            has_smiles = pd.notna(result['SMILES']) and result['SMILES']
            has_cp = pd.notna(result['Cp_solid']) or pd.notna(result['Cp_liquid']) or pd.notna(result['Cp_gas'])
            
            if has_smiles and has_cp:
                dataset.append(result)
                valid_count += 1
                
                if valid_count % 100 == 0:
                    print(f"*** Collected {valid_count} valid ML-ready rows ***")

    # Save step1 Excel (row 2 contains units)
    df = pd.DataFrame(dataset, columns=OUTPUT_COLUMNS)
    units_row = {
        'name': '',
        'CAS': '',
        'SMILES': '',
        'melting point': '°C',
        'boiling point': '°C',
        'Cp_solid': 'J/mol K',
        'Cp_liquid': 'J/mol K',
        'Cp_gas': 'J/mol K',
        'phase_at_25°C': ''
    }
    df_with_units = pd.concat([pd.DataFrame([units_row]), df], ignore_index=True)

    base_name = OUTPUT_FILE.rsplit('.', 1)[0]
    step1_file = f"{base_name}_step1.xlsx"
    step2_file = f"{base_name}_step2.xlsx"

    df_with_units.to_excel(step1_file, index=False)

    # Build step2 with RDKit descriptors from SMILES.
    if Chem is None:
        print("Warning: RDKit is not available. MW and rotatable_bonds will be empty.")

    rdkit_features = df['SMILES'].apply(get_rdkit_features)
    df['MW'] = rdkit_features.apply(lambda x: x[0])
    df['rotatable_bonds'] = rdkit_features.apply(lambda x: x[1])
    df['canonical_smiles'] = rdkit_features.apply(lambda x: x[2])

    # Remove physically inconsistent rows before step2 export (BP must be >= MP when both exist).
    invalid_mp_bp_mask = (
        pd.notna(df['melting point'])
        & pd.notna(df['boiling point'])
        & (df['boiling point'] < df['melting point'])
    )
    invalid_mp_bp_count = int(invalid_mp_bp_mask.sum())
    if invalid_mp_bp_count > 0:
        print(f"\nRemoving {invalid_mp_bp_count} rows where boiling point < melting point before step2 export.")
    df = df[~invalid_mp_bp_mask].copy()

    df_step2 = df[STEP2_OUTPUT_COLUMNS]

    step2_units_row = {
        'name': '',
        'CAS': '',
        'SMILES': '',
        'canonical_smiles': '',
        'MW': 'g/mol',
        'rotatable_bonds': '',
        'melting point': '°C',
        'boiling point': '°C',
        'Cp_solid': 'J/mol K',
        'Cp_liquid': 'J/mol K',
        'Cp_gas': 'J/mol K',
        'phase_at_25°C': ''
    }
    df_step2_with_units = pd.concat([pd.DataFrame([step2_units_row]), df_step2], ignore_index=True)
    df_step2_with_units.to_excel(step2_file, index=False)

    # Save step2_strict: keep only compounds where phase_at_25°C matches corresponding Cp value
    print("\nCreating step2 strict-filtered versions (phase must match Cp)...")
    has_mp_bp = pd.notna(df['melting point']) & pd.notna(df['boiling point'])
    mask_solid = has_mp_bp & (df['phase_at_25°C'] == 'solid') & (pd.notna(df['Cp_solid']))
    mask_liquid = has_mp_bp & (df['phase_at_25°C'] == 'liquid') & (pd.notna(df['Cp_liquid']))
    mask_gas = has_mp_bp & (df['phase_at_25°C'] == 'gas') & (pd.notna(df['Cp_gas']))
    
    df_strict = df[mask_solid | mask_liquid | mask_gas].copy()
    
    if len(df_strict) > 0:
        # Save step2_strict
        step2_strict_file = f"{base_name}_step2_strict.xlsx"
        df_strict_step2 = df_strict[STEP2_OUTPUT_COLUMNS]
        strict_units_row_step2 = {
            'name': '',
            'CAS': '',
            'SMILES': '',
            'canonical_smiles': '',
            'MW': 'g/mol',
            'rotatable_bonds': '',
            'melting point': '°C',
            'boiling point': '°C',
            'Cp_solid': 'J/mol K',
            'Cp_liquid': 'J/mol K',
            'Cp_gas': 'J/mol K',
            'phase_at_25°C': ''
        }
        df_strict_with_units = pd.concat([pd.DataFrame([strict_units_row_step2]), df_strict_step2], ignore_index=True)
        df_strict_with_units.to_excel(step2_strict_file, index=False)
        print(f"  Saved {len(df_strict)} compounds to '{step2_strict_file}'.")
        strict_summary = f"Strict version created with {len(df_strict)} compounds to '{step2_strict_file}'."
    else:
        print("  No compounds passed strict filtering.")
        strict_summary = "Strict version was not created because no compounds passed strict filtering."

    print(f"\nExtraction Complete! Processed all {total} CAS candidates. Saved {len(df)} compounds that met criteria to '{step1_file}', '{step2_file}'.")
    print(strict_summary)

if __name__ == "__main__":
    start_time = time.time()
    # Handle Windows asyncio event loop issue
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")