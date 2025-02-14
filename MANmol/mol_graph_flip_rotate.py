import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import pickle
import time
from joblib import Parallel, delayed 

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        atom_feature = [
            atom.GetAtomicNum(), 
            atom.GetTotalNumHs(), 
            int(atom.GetIsAromatic()) 
        ]
        node_features.append(atom_feature)

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_features.append([i, j, bond_type])

    edge_index = torch.tensor(np.array(edge_features)[:, :2], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_features)[:, 2:], dtype=torch.float) 

    return node_features, edge_index, edge_attr

def get_rotation_matrix(axis, angle=90):
    theta = np.radians(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def get_flip_matrix(axis):
    if axis == 'x':
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis == 'y':
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'z':
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

def transform_molecule_graph(smiles, transformation, timeout=500):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    start_time = time.time()

    try:
        if AllChem.EmbedMolecule(mol, maxAttempts=10000) != 0:
            return None 

        if time.time() - start_time > timeout:
            return None

        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()

        for atom_idx in range(mol.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(atom_idx))
            new_pos = np.dot(transformation, pos)
            conf.SetAtomPosition(atom_idx, new_pos)

        return smiles_to_graph(Chem.MolToSmiles(mol))
    except Exception as e:
        print(f"Error processing molecule with SMILES {smiles}: {e}")
        return None

def process_molecule(row):
    smiles = row['smiles']
    original_graph = smiles_to_graph(smiles)

    if original_graph is None:
        return None

    original_dict = row.to_dict()
    original_dict['graph'] = original_graph

    graph_data = [original_dict]

    transformations = {
        'rotated_x': get_rotation_matrix('x'),
        'rotated_y': get_rotation_matrix('y'),
        'rotated_z': get_rotation_matrix('z'),
        'flipped_x': get_flip_matrix('x'),
        'flipped_y': get_flip_matrix('y'),
        'flipped_z': get_flip_matrix('z')
    }

    for key, transformation in transformations.items():
        transformed_graph = transform_molecule_graph(smiles, transformation)

        if transformed_graph is not None:
            transformed_dict = row.to_dict()
            transformed_dict['graph'] = transformed_graph
            graph_data.append(transformed_dict)

    return graph_data

df = pd.read_csv("./data/all_porperty_AE_data.csv")

results = Parallel(n_jobs=44)(delayed(process_molecule)(row) for _, row in df.iterrows())

graph_data_list = [item for sublist in results if sublist is not None for item in sublist]

with open('./data/smiles_graph_flip_rotate/molecule_graphs-flip_rotate.pkl', 'wb') as f:
    pickle.dump(graph_data_list, f)
print(len(graph_data_list))

print("The data has been successfully saved to './data/smiles_graph_flip_rotate/molecule_graphs.pkl'")
