import pandas as pd
import pickle
from tqdm import tqdm
import torch 

with open('./data/smiles_graph_flip_rotate/molecule_graphs-flip_rotate.pkl', 'rb') as f:
    molecule_graphs = pickle.load(f)     

molecule_graphs_dict = {}
for entry in molecule_graphs:
    mol_id = entry['ID']
    if mol_id not in molecule_graphs_dict:
        molecule_graphs_dict[mol_id] = []
    molecule_graphs_dict[mol_id].append(entry)

output_file = './data/14.5M-all_data_enhancement-smi_des_graph.pkl'

chunksize = 1000000  

total_enhanced = 0

with open(output_file, 'wb') as f_out: 
    smiles_iter = pd.read_csv('./data/smiles_enumeration/2.11M-all_AE_smiles_random_order-ID.csv', chunksize=chunksize, on_bad_lines='skip')
    
    for smiles_chunk in tqdm(smiles_iter, desc="Processing chunks"):
        all_data_enhanced = []
        for _, row in smiles_chunk.iterrows():
            smiles_id = row['ID']
            smiles = row['smiles']

            if smiles_id in molecule_graphs_dict:
                for graph_entry in molecule_graphs_dict[smiles_id]:
                    new_entry = graph_entry.copy()
                    new_entry['smiles'] = smiles 
                    all_data_enhanced.append(new_entry)
        
        pickle.dump(all_data_enhanced, f_out)
        total_enhanced += len(all_data_enhanced) 
