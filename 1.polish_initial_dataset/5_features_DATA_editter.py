import pandas as pd
import requests
import time
import re
import numpy as np
import math
from urllib.parse import quote
from difflib import SequenceMatcher
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

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
    return selected

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

    # Even when confidence is low, return the highest-similarity candidate.
    if scored:
        return scored[0][0]

    return None

def prompt_cid_selection(name, candidates):
    """Auto-pick CID with highest synonym similarity (no user input)."""
    target = normalize_name(name)
    scored = []

    for cid in candidates:
        synonyms = get_synonyms_for_cid(cid)
        if not synonyms:
            scored.append((cid, 0.0))
            continue

        best_ratio = 0.0
        for syn in synonyms:
            ratio = SequenceMatcher(None, target, normalize_name(syn)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
        scored.append((cid, best_ratio))

    # Deterministic tie-breaker: smaller CID first.
    scored.sort(key=lambda x: (-x[1], x[0]))
    selected_cid = scored[0][0] if scored else None

    if selected_cid is not None:
        print(f"    -> [CID AUTO] '{name}' 후보 {len(candidates)}개 중 CID {selected_cid} 자동 선택")

    return selected_cid

def get_computed_properties(cid):
    """Deprecated: kept for backward compatibility but no longer used."""
    return None

def get_rdkit_properties_from_smiles(smiles_text):
    """Calculate canonical SMILES, MW, and rotatable bonds from SMILES using RDKit."""
    if smiles_text is None:
        return None

    smiles = str(smiles_text).strip()
    if smiles == '' or smiles == '-':
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        canonical = Chem.MolToSmiles(mol, canonical=True)
        return {
            'canonical_smiles': canonical,
            'molecular_weight': round(float(Descriptors.MolWt(mol)), 4),
            'rotatable_bonds': int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        }
    except Exception:
        return None

def check_and_update_text(df, index, col_name, new_val):
    """Compare and update dataframe text field with normalization."""
    if new_val is None:
        return

    new_text = str(new_val).strip()
    if new_text == '' or new_text == '-':
        return

    existing_val = df.at[index, col_name]
    if pd.isna(existing_val) or str(existing_val).strip() in ['', '-']:
        df.at[index, col_name] = new_text
        print(f"    -> [FILLED] {col_name}: {new_text}")
        return

    exist_text = str(existing_val).strip()
    if exist_text != new_text:
        df.at[index, col_name] = new_text
        print(f"    -> [CORRECTED] {col_name}: {exist_text} -> {new_text}")

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
print("Starting PubChem Verification Pipeline...\n")

df = pd.read_excel(INPUT_FILE)

TEXT_COLUMNS = [
    "name",
    "chemical_compound_name",
    "compound_name",
    "smiles",
    "canonical_smiles",
    "pubchem_name",
    "iupac_name",
    "molecular_formula",
    "verification_status",
    "source",
    "note",
    "error_message",
]

df = df.replace('-', np.nan)

# Ensure canonical_smiles exists right next to smiles when possible
if 'canonical_smiles' not in df.columns:
    if 'smiles' in df.columns:
        insert_at = df.columns.get_loc('smiles') + 1
        df.insert(insert_at, 'canonical_smiles', np.nan)
    else:
        df['canonical_smiles'] = np.nan

# Convert text columns to object dtype
for col in TEXT_COLUMNS:
    if col in df.columns:
        df[col] = df[col].astype("object")

# Ensure target columns exist
features = [
    'molecular_weight',
    'rotatable_bonds',
    'melting_point',
    'boiling_point',
    'critical_temperature',
    'acentric_factor'
]

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

    # Part A: Use SMILES + RDKit for canonical SMILES, MW, Rotatable Bonds
    structure = get_structure_for_cid(cid)
    pubchem_smiles = structure.get('isomeric_smiles') if structure else None

    source_smiles = row['smiles'] if 'smiles' in df.columns and pd.notna(row.get('smiles')) else pubchem_smiles
    if (source_smiles is None or str(source_smiles).strip() == '' or str(source_smiles).strip() == '-') and pubchem_smiles:
        source_smiles = pubchem_smiles

    if 'smiles' in df.columns and pubchem_smiles:
        if pd.isna(df.at[index, 'smiles']) or str(df.at[index, 'smiles']).strip() == '-':
            df.at[index, 'smiles'] = pubchem_smiles
            print(f"    -> [FILLED] smiles: {pubchem_smiles}")

    rdkit_props = get_rdkit_properties_from_smiles(source_smiles)
    if rdkit_props:
        check_and_update_text(df, index, 'canonical_smiles', rdkit_props['canonical_smiles'])
        # MW follows previous 1% tolerance policy.
        check_and_update(df, index, 'molecular_weight', rdkit_props['molecular_weight'], rel_tol=0.01)
        # Rotatable bonds are integer descriptor, so use near-exact comparison.
        check_and_update(df, index, 'rotatable_bonds', rdkit_props['rotatable_bonds'], abs_tol=0.1)
    else:
        print("    -> [RDKit] SMILES 파싱 실패로 MW/rotatable_bonds 비교 생략")

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

