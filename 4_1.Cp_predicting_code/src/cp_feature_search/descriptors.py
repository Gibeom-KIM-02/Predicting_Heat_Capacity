from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors


def parse_smiles_with_status(smiles: Any) -> Tuple[Optional[Chem.Mol], str, str, Optional[str]]:
    """
    Parse SMILES quietly and return:
        mol, status, message, canonical_smiles
    """
    if pd.isna(smiles):
        return None, "empty", "NaN SMILES", None

    smi = str(smiles).strip()
    if not smi:
        return None, "empty", "blank SMILES", None

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smi, sanitize=False)

        if mol is None:
            return None, "parse_failed", "MolFromSmiles returned None", None

        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
        except Exception as exc:
            return None, "sanitize_failed", str(exc), None

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mol, "ok", "", canonical

    except Exception as exc:
        return None, "exception", str(exc), None


def canonicalize_smiles(smiles: Any) -> tuple[Optional[str], bool]:
    mol, status, _, canonical = parse_smiles_with_status(smiles)
    if status != "ok" or mol is None or canonical is None:
        return None, False
    return canonical, True


def _atom_count(mol: Chem.Mol, atomic_numbers: set[int]) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in atomic_numbers)


def calc_basic_rdkit_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Small interpretable descriptor group.
    Good for explaining final feature choices.
    """
    heavy = mol.GetNumHeavyAtoms()
    rot = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    carbon = _atom_count(mol, {6})
    halogen = _atom_count(mol, {9, 17, 35, 53})

    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "HeavyAtomCount": float(heavy),
        "NumValenceElectrons": float(Descriptors.NumValenceElectrons(mol)),
        "NumRotatableBonds": float(rot),
        "RingCount": float(rings),
        "NumAromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "NumAliphaticRings": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "BalabanJ": float(Descriptors.BalabanJ(mol)),
        "Kappa1": float(Descriptors.Kappa1(mol)),
        "Kappa2": float(Descriptors.Kappa2(mol)),
        "Kappa3": float(Descriptors.Kappa3(mol)),
        "HeteroAtomCount": float(hetero),
        "HalogenCount": float(halogen),
        "FormalCharge": float(Chem.GetFormalCharge(mol)),
        "RotPerHeavyAtom": float(rot / heavy) if heavy else np.nan,
        "RingPerHeavyAtom": float(rings / heavy) if heavy else np.nan,
        "HeteroAtomFraction": float(hetero / heavy) if heavy else np.nan,
        "CarbonFraction": float(carbon / heavy) if heavy else np.nan,
    }


def calc_rdkit_small_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Slightly expanded RDKit descriptor group.
    Still interpretable, but larger than basic_2.
    """
    d = calc_basic_rdkit_descriptors(mol)
    d.update(
        {
            "MolMR": float(Crippen.MolMR(mol)),
            "NHOHCount": float(Lipinski.NHOHCount(mol)),
            "NOCount": float(Lipinski.NOCount(mol)),
            "NumSaturatedRings": float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
            "NumHeterocycles": float(rdMolDescriptors.CalcNumHeterocycles(mol)),
            "NumAmideBonds": float(rdMolDescriptors.CalcNumAmideBonds(mol)),
            "NumBridgeheadAtoms": float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            "NumSpiroAtoms": float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        }
    )
    return d

def _morgan_bits(mol: Chem.Mol, radius: int, n_bits: int) -> Dict[str, int]:
    """
    Calculate Morgan fingerprint bits using the modern RDKit MorganGenerator API.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    fp = generator.GetFingerprint(mol)

    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return {f"morgan_{i}": int(v) for i, v in enumerate(arr)}


def _maccs_bits(mol: Chem.Mol) -> Dict[str, int]:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # MACCS has bit 0 unused in many conventions, but keep all bits for consistency.
    return {f"maccs_{i}": int(v) for i, v in enumerate(arr)}


def featurize_dataframe(df: pd.DataFrame, feature_set: str, config: Dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """
    Add descriptors/fingerprints to df.

    Supported feature_set:
        basic_2
        rdkit_selected
        rdkit_small
        morgan
        maccs
    """
    rows = []
    meta_rows = []

    for _, row in df.iterrows():
        mol, status, message, canonical = parse_smiles_with_status(row["smiles_raw"])

        meta = row.to_dict()
        meta["parse_status"] = status
        meta["parse_message"] = message
        meta["canonical_smiles"] = canonical
        meta["valid_smiles"] = int(status == "ok")
        meta_rows.append(meta)

        if mol is None:
            rows.append({})
            continue

        if feature_set == "basic_2":
            d = calc_basic_rdkit_descriptors(mol)
            desc = {
                "MolWt": d["MolWt"],
                "NumRotatableBonds": d["NumRotatableBonds"],
            }

        elif feature_set == "rdkit_selected":
            desc = calc_basic_rdkit_descriptors(mol)

        elif feature_set == "rdkit_small":
            desc = calc_rdkit_small_descriptors(mol)

        elif feature_set == "morgan":
            morgan_cfg = config.get("morgan", {})
            radius = int(morgan_cfg.get("radius", 2))
            n_bits = int(morgan_cfg.get("n_bits", 1024))
            desc = _morgan_bits(mol, radius=radius, n_bits=n_bits)

        elif feature_set == "maccs":
            desc = _maccs_bits(mol)

        else:
            raise ValueError(f"Unsupported feature_set={feature_set}")

        rows.append(desc)

    meta_df = pd.DataFrame(meta_rows)
    X = pd.DataFrame(rows)

    # Drop invalid SMILES rows.
    valid_mask = meta_df["valid_smiles"].eq(1)
    meta_df = meta_df[valid_mask].reset_index(drop=True)
    X = X[valid_mask].reset_index(drop=True)

    # Ensure numeric and stable column order.
    X = X.apply(pd.to_numeric, errors="coerce")
    feature_names = list(X.columns)

    out = pd.concat([meta_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    out = out.dropna(subset=["Cp_J_molK"]).copy()

    return out, feature_names
