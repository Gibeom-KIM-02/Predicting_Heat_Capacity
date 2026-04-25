from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .feature_registry import get_feature_names


def safe_mol(smiles: Any):
    """Convert SMILES to RDKit Mol. Return None if invalid."""
    mol, status, message, canonical = parse_smiles_with_status(smiles)
    return mol

def canonicalize_smiles(smiles: Any) -> tuple[Optional[str], bool]:
    """Return canonical isomeric SMILES and validity flag."""
    mol, status, message, canonical = parse_smiles_with_status(smiles)
    if status != "ok" or canonical is None:
        return None, False
    return canonical, True


def parse_smiles_with_status(smiles):
    """
    Parse SMILES quietly and return:
        mol, status, message, canonical_smiles

    status:
        ok
        empty
        parse_failed
        sanitize_failed
        exception
    """
    if pd.isna(smiles):
        return None, "empty", "NaN SMILES", None

    smi = str(smiles).strip()
    if not smi:
        return None, "empty", "blank SMILES", None

    try:
        # Block RDKit console messages during parsing
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smi, sanitize=False)

        if mol is None:
            return None, "parse_failed", "MolFromSmiles returned None", None

        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
        except Exception as e:
            return None, "sanitize_failed", str(e), None

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mol, "ok", "", canonical

    except Exception as e:
        return None, "exception", str(e), None


def atom_count(mol, atomic_numbers: set[int]) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in atomic_numbers)


def calc_selected_rdkit_descriptors(smiles: Any) -> Optional[Dict[str, float]]:
    """Calculate the selected interpretable RDKit descriptors."""
    mol = safe_mol(smiles)
    if mol is None:
        return None

    heavy = mol.GetNumHeavyAtoms()
    carbon = atom_count(mol, {6})
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    halogen = atom_count(mol, {9, 17, 35, 53})
    rot = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)

    d = {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "HeavyAtomCount": heavy,
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NumRotatableBonds": rot,
        "RingCount": rings,
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        "HeteroAtomCount": hetero,
        "HalogenCount": halogen,
        "FormalCharge": Chem.GetFormalCharge(mol),
        "RotPerHeavyAtom": rot / heavy if heavy else np.nan,
        "RingPerHeavyAtom": rings / heavy if heavy else np.nan,
        "HeteroAtomFraction": hetero / heavy if heavy else np.nan,
        "CarbonFraction": carbon / heavy if heavy else np.nan,
    }
    return d


def calc_rdkit_small_descriptors(smiles: Any) -> Optional[Dict[str, float]]:
    """
    Calculate a slightly expanded RDKit descriptor set.

    This includes all selected_rdkit_v1 descriptors plus additional
    chemically interpretable RDKit descriptors.
    """
    base = calc_selected_rdkit_descriptors(smiles)
    if base is None:
        return None

    mol = safe_mol(smiles)
    if mol is None:
        return None

    extra = {
        "MolMR": Crippen.MolMR(mol),
        "NHOHCount": Lipinski.NHOHCount(mol),
        "NOCount": Lipinski.NOCount(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumHeterocycles": rdMolDescriptors.CalcNumHeterocycles(mol),
        "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
    }

    base.update(extra)
    return base


def featurize_smiles_list(smiles_list: list[Any], feature_set: str) -> pd.DataFrame:
    """Featurize a list of SMILES into a DataFrame with fixed feature order."""
    feature_names = get_feature_names(feature_set)
    rows = []

    selected_like_sets = {
        "selected_rdkit_v1",
        "shap_top10_v1",
        "shap_top15_v1",
        "chem_compact_v1",
    }

    for smi in smiles_list:
        if feature_set in selected_like_sets:
            desc = calc_selected_rdkit_descriptors(smi)

        elif feature_set == "rdkit_small":
            desc = calc_rdkit_small_descriptors(smi)

        else:
            raise ValueError(f"Unsupported feature_set={feature_set}")

        rows.append(
            desc if desc is not None else {name: np.nan for name in feature_names}
        )

    return pd.DataFrame(rows)[feature_names]


def featurize_one_smiles(smiles: str, feature_set: str) -> pd.DataFrame:
    """Featurize a single SMILES and return a one-row DataFrame."""
    return featurize_smiles_list([smiles], feature_set=feature_set)
