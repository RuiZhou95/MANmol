import pandas as pd
from rdkit import Chem
from tqdm import tqdm


df = pd.read_csv('./data/all_porperty_AE_data.csv')


def generate_all_smiles_variants(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    variants = set()
    
    for _ in range(80000): 
        random_smiles = Chem.MolToSmiles(mol, doRandom=True, kekuleSmiles=False)
        variants.add(random_smiles)
    
    return list(variants)

new_data = []

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing SMILES"):
    smiles = row['smiles']
    ID = row['ID']
    smiles_variants = generate_all_smiles_variants(smiles)
    
    for variant in smiles_variants:
        new_data.append([ID, variant])

new_df = pd.DataFrame(new_data, columns=['ID', 'smiles'])
new_df.to_csv('./data/A.csv', index=False)

A_df = pd.read_csv('./data/A.csv')

df_to_merge = df.drop(columns=['ID', 'smiles'])

final_data = []

for _, row in A_df.iterrows():
    ID = row['ID']
    smiles = row['smiles']
    
    matching_row = df[df['ID'] == ID].iloc[0]
    
    additional_data = matching_row[2:].values 
    
    final_data.append([ID, smiles] + list(additional_data))

final_df = pd.DataFrame(final_data, columns=['ID', 'smiles'] + list(df.columns[2:]))
final_df.to_csv('./data/all_AE_data_merged.csv', index=False)
