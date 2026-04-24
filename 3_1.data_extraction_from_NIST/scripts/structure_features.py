from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .utils import normalize_text


def configure_rdkit_logging(rdkit_log_level: str) -> None:
    level = str(rdkit_log_level).strip().lower()
    if level == "none":
        RDLogger.DisableLog("rdApp.debug")
        RDLogger.DisableLog("rdApp.info")
        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
    elif level == "warning":
        RDLogger.DisableLog("rdApp.warning")
    elif level == "error":
        RDLogger.DisableLog("rdApp.warning")
    elif level == "keep":
        pass


def mol_from_compound(comp: dict[str, Any]):
    inchi = normalize_text(comp.get("inchi", ""))
    if inchi:
        try:
            mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=True)
            if mol is not None:
                return mol, "inchi"
        except Exception:
            return None, "none"
    return None, "none"


def compute_rdkit_features(mol: Chem.Mol) -> dict[str, Any]:
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "H_bond_donors": int(Lipinski.NumHDonors(mol)),
        "H_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "logP": float(Crippen.MolLogP(mol)),
    }
