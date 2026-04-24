import pandas as pd
import requests
import time
import re
import numpy as np
import math
from urllib.parse import quote
from difflib import SequenceMatcher

# Configuration
DELAY = 0.3 
CID_LOOKUP_RETRIES = 3
INPUT_FILE = 'mid_project_sample_data.xlsx'
OUTPUT_FILE = 'mid_project_5features_verified.xlsx'

SYNONYM_CACHE = {}
STRUCTURE_CACHE = {}

# ==========================================
# 1. PubChem API Helper Functions
# ==========================================
def get_cid_from_name(name, retries=CID_LOOKUP_RETRIES):
    """Fetch CID using name -> candidate list -> synonym validation."""
    candidates = get_cid_candidates_from_name(name, retries=retries)
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    selected = select_best_cid(name, candidates)
    if selected is not None:
        return selected

    return prompt_cid_selection(name, candidates)

def get_cid_candidates_from_name(name, retries=CID_LOOKUP_RETRIES):
    """Fetch possible CID candidates for a compound name."""
    encoded = quote(str(name))
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/JSON"
    for _ in range(retries):
        try:
            res = requests.get(url, timeout=10)
            time.sleep(DELAY)
            if res.status_code == 200:
                return res.json().get('IdentifierList', {}).get('CID', [])
        except Exception:
            pass
    return []

def normalize_name(s):
    """Normalize name for robust exact-ish matching."""
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def get_synonyms_for_cid(cid):
    """Get synonym list for CID and cache it."""
    if cid in SYNONYM_CACHE:
        return SYNONYM_CACHE[cid]

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    synonyms = []
    try:
        res = requests.get(url, timeout=10)
        time.sleep(DELAY)
        if res.status_code == 200:
            info = res.json().get('InformationList', {}).get('Information', [])
            if info:
                synonyms = info[0].get('Synonym', [])
    except Exception:
        pass

    SYNONYM_CACHE[cid] = synonyms
    return synonyms

def get_structure_for_cid(cid):
    """Get structural identifiers for a CID and cache them."""
    if cid in STRUCTURE_CACHE:
        return STRUCTURE_CACHE[cid]

    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
        f"/property/IsomericSMILES,InChIKey,IUPACName,MolecularFormula/JSON"
    )
    structure = {'isomeric_smiles': None, 'inchi_key': None, 'iupac_name': None, 'formula': None}

    try:
        res = requests.get(url, timeout=10)
        time.sleep(DELAY)
        if res.status_code == 200:
            props = res.json().get('PropertyTable', {}).get('Properties', [])
            if props:
                item = props[0]
                structure = {
                    'isomeric_smiles': item.get('IsomericSMILES'),
                    'inchi_key': item.get('InChIKey'),
                    'iupac_name': item.get('IUPACName'),
                    'formula': item.get('MolecularFormula'),
                }
    except Exception:
        pass

    STRUCTURE_CACHE[cid] = structure
    return structure

def select_best_cid(name, candidates):
    """Choose best CID by name first, then structural identifiers when needed."""
    target = normalize_name(name)
    scored = []
    exact_matches = []

    for cid in candidates:
        synonyms = get_synonyms_for_cid(cid)
        if not synonyms:
            scored.append((cid, 0.0, ''))
            continue

        norm_synonyms = [normalize_name(s) for s in synonyms]
        if target in norm_synonyms:
            exact_matches.append(cid)

        best_ratio = 0.0
        best_syn = ''
        for syn, norm_syn in zip(synonyms, norm_synonyms):
            ratio = SequenceMatcher(None, target, norm_syn).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_syn = syn
        scored.append((cid, best_ratio, best_syn))

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        exact_structures = [get_structure_for_cid(cid) for cid in exact_matches]
        unique_inchi = {item.get('inchi_key') for item in exact_structures if item.get('inchi_key')}
        if len(unique_inchi) == 1:
            return exact_matches[0]

        return prompt_cid_selection(name, exact_matches)

    scored.sort(key=lambda x: x[1], reverse=True)

    if scored and scored[0][1] >= 0.92:
        top_score = scored[0][1]
        top_candidates = [cid for cid, ratio, _ in scored if ratio == top_score]
        if len(top_candidates) == 1:
            return top_candidates[0]

        top_structures = [get_structure_for_cid(cid) for cid in top_candidates]
        unique_inchi = {item.get('inchi_key') for item in top_structures if item.get('inchi_key')}
        if len(unique_inchi) == 1:
            return top_candidates[0]

        return prompt_cid_selection(name, top_candidates)

    return None

def prompt_cid_selection(name, candidates):
    """Prompt user to choose CID if multiple ambiguous candidates exist."""
    print(f"    -> [CID 모호] '{name}' 후보가 {len(candidates)}개입니다.")
    for i, cid in enumerate(candidates[:10], start=1):
        synonyms = get_synonyms_for_cid(cid)
        label = synonyms[0] if synonyms else 'N/A'
        structure = get_structure_for_cid(cid)
        smiles = structure.get('isomeric_smiles') or 'N/A'
        inchikey = structure.get('inchi_key') or 'N/A'
        print(f"       {i}) CID {cid} | {label}")
        print(f"          SMILES: {smiles}")
        print(f"          InChIKey: {inchikey}")

    while True:
        user_input = input("       선택 번호 입력 (Enter=skip): ").strip()
        if user_input == '':
            return None
        if user_input.isdigit():
            idx = int(user_input)
            if 1 <= idx <= min(len(candidates), 10):
                return candidates[idx - 1]
        print("       -> [입력 오류] 표시된 번호를 입력하세요.")

def get_computed_properties(cid):
    """ Fetch computed properties (MW, Rotatable Bonds) via REST API """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularWeight,RotatableBondCount/JSON"
    try:
        res = requests.get(url, timeout=10)
        time.sleep(DELAY)
        if res.status_code == 200:
            props = res.json()['PropertyTable']['Properties'][0]
            return {
                'molecular_weight': props.get('MolecularWeight'),
                'rotatable_bonds': props.get('RotatableBondCount')
            }
    except:
        pass
    return None

def extract_recursive(node, target_heading, results):
    """ Recursive search in PUG View JSON """
    if isinstance(node, dict):
        if node.get('TOCHeading') == target_heading:
            for info in node.get('Information', []):
                for val in info.get('Value', {}).get('StringWithMarkup', []):
                    results.append(val['String'])
        for key, val in node.items():
            extract_recursive(val, target_heading, results)
    elif isinstance(node, list):
        for item in node:
            extract_recursive(item, target_heading, results)

def get_experimental_texts(cid, heading):
    """ Fetch experimental text data via PUG View API """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    results = []
    try:
        res = requests.get(url, timeout=10)
        time.sleep(DELAY)
        if res.status_code == 200:
            extract_recursive(res.json().get('Record', {}).get('Section', []), heading, results)
    except:
        pass
    return results

def clean_temperature_text(texts):
    """Extract Celsius from explicit C/F/K values only."""
    for text in texts:
        # Range with unit: e.g., 122 to 126 F
        for m in re.finditer(r"([-+]?\d*\.?\d+)\s*(?:to|-|~)\s*([-+]?\d*\.?\d+)\s*(?:°|deg)?\s*([CFK])\b", text, re.IGNORECASE):
            a = float(m.group(1))
            b = float(m.group(2))
            unit = m.group(3).upper()
            mid = (a + b) / 2.0
            if unit == 'C':
                c_val = mid
            elif unit == 'F':
                c_val = (mid - 32.0) * 5.0 / 9.0
            else:
                c_val = mid - 273.15
            if -250 <= c_val <= 1000:
                return round(c_val, 2)

        # Single value with unit
        for m in re.finditer(r"([-+]?\d*\.?\d+)\s*(?:°|deg)?\s*([CFK])\b", text, re.IGNORECASE):
            raw = float(m.group(1))
            unit = m.group(2).upper()
            if unit == 'C':
                c_val = raw
            elif unit == 'F':
                c_val = (raw - 32.0) * 5.0 / 9.0
            else:
                c_val = raw - 273.15
            if -250 <= c_val <= 1000:
                return round(c_val, 2)

    return None

def check_and_update(df, index, col_name, new_val, abs_tol=0.0, rel_tol=0.0):
    """ Compare and update dataframe with tolerance check """
    if new_val is None:
        return

    try:
        new_float = float(new_val)
    except (ValueError, TypeError):
        return
        
    existing_val = df.at[index, col_name]
    
    # Empty cell -> Fill directly
    if pd.isna(existing_val) or str(existing_val).strip() == '-':
        df.at[index, col_name] = new_float
        print(f"    -> [FILLED] {col_name}: {new_float}")
        return

    # Existing cell -> Compare and correct
    try:
        exist_float = float(str(existing_val).replace('*', '').strip())
        is_match = False
        
        if abs_tol > 0:
            is_match = math.isclose(exist_float, new_float, abs_tol=abs_tol)
        else:
            is_match = math.isclose(exist_float, new_float, rel_tol=rel_tol)
            
        if not is_match:
            df.at[index, col_name] = new_float
            print(f"    -> [CORRECTED] {col_name}: {exist_float} -> {new_float}")
            
    except ValueError:
        # If existing value is broken text, overwrite
        df.at[index, col_name] = new_float
        print(f"    -> [RECOVERED] {col_name}: {new_float}")

# ==========================================
# 2. Main Pipeline
# ==========================================
print("Starting PubChem 6-Feature Verification Pipeline...\n")

df = pd.read_excel(INPUT_FILE)
df = df.replace('-', np.nan)

# Ensure target columns exist
features = ['molecular_weight', 'rotatable_bonds', 'melting_point', 'boiling_point', 'critical_temperature', 'acentric_factor']
for col in features:
    if col not in df.columns:
        df[col] = np.nan

for index, row in df.iterrows():
    name = row['name']
    if pd.isna(name) or name == '-': 
        continue
        
    print(f"[{index}] Processing {name}...")
    cid = get_cid_from_name(name)
    
    if not cid:
        print("    -> CID NOT FOUND")
        continue

    # Part A: Computed Properties (MW, Rotatable Bonds) - High precision, 1% tolerance
    computed = get_computed_properties(cid)
    if computed:
        check_and_update(df, index, 'molecular_weight', computed['molecular_weight'], rel_tol=0.01)
        # Rotatable bonds must be exact integer match, so tolerance is effectively 0
        check_and_update(df, index, 'rotatable_bonds', computed['rotatable_bonds'], abs_tol=0.1)

    # Part B: Experimental Temperatures (MP, BP, Tc) - 3.0°C tolerance
    # Melting Point
    mp_texts = get_experimental_texts(cid, 'Melting Point')
    mp_val = clean_temperature_text(mp_texts)
    check_and_update(df, index, 'melting_point', mp_val, abs_tol=3.0)

    # Boiling Point
    bp_texts = get_experimental_texts(cid, 'Boiling Point')
    bp_val = clean_temperature_text(bp_texts)
    check_and_update(df, index, 'boiling_point', bp_val, abs_tol=3.0)
    
    # Critical Temperature (Often sparse in PubChem)
    tc_texts = get_experimental_texts(cid, 'Critical Temperature')
    tc_val = clean_temperature_text(tc_texts)
    check_and_update(df, index, 'critical_temperature', tc_val, abs_tol=3.0)

    # Note: Acentric Factor is typically not present in PubChem PUG View.
    # It relies on specialized DBs (like thermo), so it is skipped in the search but kept in df.

# Post-processing: fill remaining NaN with '-' to match original style
df = df.fillna('-')
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nPipeline Complete! Data saved to: {OUTPUT_FILE}")
