from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from rdkit import Chem


COMMON_NAME_TO_SMILES = {
    "acetone": "CC(=O)C",
    "propanone": "CC(=O)C",
    "water": "O",
    "methane": "C",
    "ethane": "CC",
    "propane": "CCC",
    "methanol": "CO",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "acetic acid": "CC(=O)O",
    "formic acid": "O=CO",
}


def _canonicalize_smiles_text(smiles: str | None) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit."""
    if smiles is None or str(smiles).strip() == "":
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _resolve_with_pubchempy(name: str) -> Optional[str]:
    """
    Resolve name to SMILES using PubChemPy.

    Some PubChemPy versions may return a compound object but leave
    canonical_smiles / isomeric_smiles as None. In that case this returns None,
    and the caller should fall back to PUG-REST.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        return None

    try:
        compounds = pcp.get_compounds(name, namespace="name")
    except Exception:
        return None

    if not compounds:
        return None

    compound = compounds[0]

    smiles = (
        getattr(compound, "isomeric_smiles", None)
        or getattr(compound, "canonical_smiles", None)
        or getattr(compound, "connectivity_smiles", None)
    )

    return _canonicalize_smiles_text(smiles)


def _resolve_with_pugrest(name: str, timeout: int = 10) -> Optional[str]:
    """
    Resolve name to SMILES using PubChem PUG-REST directly.

    PubChem may return one of:
        IsomericSMILES
        CanonicalSMILES
        ConnectivitySMILES

    In your environment, CanonicalSMILES request returned ConnectivitySMILES,
    so all three keys are supported.
    """
    try:
        import requests
    except ImportError:
        return None

    encoded = quote(name)

    # Ask for all possible SMILES-like fields. Some PubChem responses use
    # ConnectivitySMILES instead of CanonicalSMILES.
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/IsomericSMILES,CanonicalSMILES,ConnectivitySMILES/JSON"
    )

    try:
        r = requests.get(url, timeout=timeout)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None

    rec = props[0]

    smiles = (
        rec.get("IsomericSMILES")
        or rec.get("CanonicalSMILES")
        or rec.get("ConnectivitySMILES")
    )

    return _canonicalize_smiles_text(smiles)


def resolve_name_to_smiles(name: str) -> Optional[str]:
    """
    Resolve compound name to canonical SMILES.

    Priority:
        1. PubChemPy
        2. PubChem PUG-REST direct request
        3. small fallback dictionary

    Returns:
        canonical RDKit SMILES or None
    """
    if name is None or str(name).strip() == "":
        return None

    query = str(name).strip()
    key = query.lower()

    resolved = _resolve_with_pubchempy(query)
    if resolved:
        return resolved

    resolved = _resolve_with_pugrest(query)
    if resolved:
        return resolved

    if key in COMMON_NAME_TO_SMILES:
        return _canonicalize_smiles_text(COMMON_NAME_TO_SMILES[key])

    return None