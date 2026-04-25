from __future__ import annotations

# Interpretable RDKit descriptor set for final presentation.
# These features are intentionally limited and chemically explainable.
SELECTED_RDKIT_V1 = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumValenceElectrons",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "TPSA",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "LabuteASA",
    "BalabanJ",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "HeteroAtomCount",
    "HalogenCount",
    "FormalCharge",
    "RotPerHeavyAtom",
    "RingPerHeavyAtom",
    "HeteroAtomFraction",
    "CarbonFraction",
]

SHAP_TOP10_V1 = [
    "NumValenceElectrons",
    "HeavyAtomCount",
    "LabuteASA",
    "Kappa1",
    "MolWt",
    "ExactMolWt",
    "NumRotatableBonds",
    "Kappa2",
    "FractionCSP3",
    "RotPerHeavyAtom",
]

SHAP_TOP15_V1 = [
    "NumValenceElectrons",
    "HeavyAtomCount",
    "LabuteASA",
    "Kappa1",
    "MolWt",
    "ExactMolWt",
    "NumRotatableBonds",
    "Kappa2",
    "FractionCSP3",
    "RotPerHeavyAtom",
    "Kappa3",
    "CarbonFraction",
    "HeteroAtomFraction",
    "MolLogP",
    "HeteroAtomCount",
]

CHEM_COMPACT_V1 = [
    "NumValenceElectrons",
    "HeavyAtomCount",
    "LabuteASA",
    "Kappa1",
    "NumRotatableBonds",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "FractionCSP3",
    "HeteroAtomFraction",
]

# Slightly expanded RDKit descriptor set.
# This matches the feature-search rdkit_small idea:
# selected interpretable descriptors + several additional structural descriptors.
RDKIT_SMALL = SELECTED_RDKIT_V1 + [
    "MolMR",
    "NHOHCount",
    "NOCount",
    "NumSaturatedRings",
    "NumHeterocycles",
    "NumAmideBonds",
    "NumBridgeheadAtoms",
    "NumSpiroAtoms",
]

FEATURE_SETS = {
    "selected_rdkit_v1": SELECTED_RDKIT_V1,
    "rdkit_small": RDKIT_SMALL,
    "shap_top10_v1": SHAP_TOP10_V1,
    "shap_top15_v1": SHAP_TOP15_V1,
    "chem_compact_v1": CHEM_COMPACT_V1,
}


def get_feature_names(feature_set: str) -> list[str]:
    """Return the selected feature names for a given feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set={feature_set}. "
            f"Available: {list(FEATURE_SETS)}"
        )
    return FEATURE_SETS[feature_set]