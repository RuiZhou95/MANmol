from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]


def one_hot(value, choices: list) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices] + [0.0 if value in choices else 1.0]


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetImplicitValence()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetMass() * 0.01),
    ] + one_hot(atom.GetHybridization(), HYBRIDIZATIONS) + one_hot(atom.GetChiralTag(), CHIRAL_TAGS)


def bond_features(bond: Chem.Bond) -> list[float]:
    return one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ] + one_hot(bond.GetStereo(), STEREO_TYPES)


def smiles_to_rich_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_indices.extend([[src, dst], [dst, src]])
        edge_attrs.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feature_dim()), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def node_feature_dim() -> int:
    return 8 + len(HYBRIDIZATIONS) + 1 + len(CHIRAL_TAGS) + 1


def edge_feature_dim() -> int:
    return len(BOND_TYPES) + 1 + 2 + len(STEREO_TYPES) + 1
