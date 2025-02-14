import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles
import numpy as np
from mordred import Calculator, descriptors
from mordred import error as mordred_error
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import pdist, squareform
from minepy import MINE
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm 
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
import torch
import pickle
import time
import warnings
warnings.filterwarnings("ignore")


file_path = "FreeSolv_SAMPL.csv" 
stage_00_output_file_path = "./data/0-processed-clean.csv"


smiles_column = "smiles"
target_column = "expt" 

stage_01_output_file_path = "./data/1-Des_Mordred.csv"
stage_02_1_output_file_path = './data/3-clean_Des.csv'

pearson_threshold = 0.3
spearman_threshold = 0.3
distance_threshold = 0.3
mic_threshold = 0.3

stage_02_2_output_file_path = './data/2-Cor_descriptor.csv'

stage_03_output_file_path = r"./data/3-Opt_XGB_descriptor.csv"
best_num_features = 33

stage_04_output_file_path = './data/4-smiles_random_order-ID.csv'
max_smi_num_list = [0, 10, 100] 

print("\n>>>>>>>>>>>>>>>>>>> 00clean-dataset.py >>>>>>>>>>>>>>")
data_00 = pd.read_csv(file_path)

def normalize_smiles(smiles):
    try:
        mol_00 = Chem.MolFromSmiles(smiles)
        return MolToSmiles(mol_00, canonical=True) if mol_00 else None
    except Exception:
        return None

if smiles_column in data_00.columns:
    data_00[smiles_column] = data_00[smiles_column].apply(normalize_smiles)


if target_column in data_00.columns and smiles_column in data_00.columns:
    result_data_00 = data_00[[smiles_column, target_column]]

    result_data_00.insert(0, "ID", range(1, len(result_data_00) + 1))

    result_data_00.to_csv(stage_00_output_file_path, index=False)
else:
    print(f"Error: Columns '{smiles_column}' or '{target_column}' not found in the dataset.")

print(f"Normalize the SMILES column, add the ID column, and you're done! >> {stage_00_output_file_path}")
    
print("\n>>>>>>>>>>>>>>>>>>> 01Des_Mordred.py >>>>>>>>>>>>>>")
def calculate_descriptors(mol_01, calc):

    try:
        descriptors_df = calc.pandas([mol_01])
        return descriptors_df
    except Exception as e:
        print(f"Error calculating full descriptors for molecule: {e}")
        return None

mols_01 = pd.read_csv(stage_00_output_file_path)

mols_01['rdmol'] = mols_01['smiles'].map(Chem.MolFromSmiles)

calc = Calculator(descriptors)

result_list = []

target_name = target_column

for idx, row in mols_01.iterrows():
    mol = row['rdmol']
    if mol is not None:
        try:
            descriptors_df = calculate_descriptors(mol, calc)
            
            if descriptors_df is not None:
                result_df_01 = pd.concat([pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']], 'target': [row[target_name]]}), descriptors_df], axis=1)
            else:
                result_df_01 = pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']], 'target': [row[target_name]]})
        except Exception as e:
            print(f"Error calculating descriptors for SMILES: {row['smiles']}. Error: {e}")
            result_df_01 = pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']], 'target': [row[target_name]]})
    else:
        print(f"Invalid SMILES: {row['smiles']}")
        result_df_01 = pd.DataFrame({'ID': [row['ID']], 'smiles': [row['smiles']], 'target': [row[target_name]]})
    
    result_list.append(result_df_01)

final_df_01 = pd.concat(result_list, ignore_index=True)

final_df_01 = final_df_01.applymap(lambda x: np.nan if isinstance(x, (mordred_error.Missing, mordred_error.Error)) else x)

final_df_01.to_csv(stage_01_output_file_path, index=False)

missing_values_count = final_df_01.isnull().any(axis=1).sum()
print(f"\nNumber of rows with missing descriptor values: {missing_values_count}")

print(f"Evaluate the Mordred descriptors, rename the single object property column name to 'target', done! >> {stage_01_output_file_path}")

print("\n>>>>>>>>>>>>>>>>>>> 02Statistical_filtering.py >>>>>>>>>>>>>>")
mordred_df = pd.read_csv(stage_01_output_file_path)

protected_columns = ['ID', 'smiles', 'target']
protected_df = mordred_df[protected_columns]

numeric_df = mordred_df.drop(columns=protected_columns).select_dtypes(include=[np.number])

error_values = (np.nan, np.inf, -np.inf)

numeric_df = numeric_df.loc[:, mordred_df.isin(error_values).sum() <= 5]

print("Initial combined data after removing columns with >5 missing/error values:")
print(f"{numeric_df.shape[0]} rows and {numeric_df.shape[1]} columns.")

numeric_df = numeric_df.apply(lambda x: np.where(np.isin(x, error_values), np.nan, x))
numeric_df.fillna(numeric_df.mean(), inplace=True)

non_zero_std = numeric_df.std() != 0
numeric_df = numeric_df.loc[:, non_zero_std]

numeric_df = numeric_df.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

mordred_df = pd.concat([protected_df, numeric_df], axis=1)

mordred_df.to_csv(stage_02_1_output_file_path, index=False)

print(f"Cleaned data has {mordred_df.shape[0]} rows and {mordred_df.shape[1]} columns.")

all_data = pd.read_csv(stage_02_1_output_file_path)

X = all_data.iloc[:, 3:] 
y = all_data.iloc[:, :3] 

print(f"Number of descriptor columns before any processing: {X.shape}")

vt = VarianceThreshold(threshold=0.1)
X_selected = vt.fit_transform(X)
lowvariance_data = pd.DataFrame(X_selected)

all_name = X.columns.values.tolist()
select_name_index0 = vt.get_support(indices=True)
select_name0 = [all_name[i] for i in select_name_index0]

lowvariance_data.columns = select_name0
lowvariance_data_y = pd.concat((y, lowvariance_data), axis=1)

print(f"Number of descriptor columns after variance filtering: {lowvariance_data.shape}") 

all_data = lowvariance_data_y
data = all_data.iloc[:, ~all_data.columns.isin(["ID", "smiles"])]
descriptor_data = data.iloc[:, data.columns != "target"]
descriptor_name_list = list(descriptor_data)

scaler = StandardScaler()
data_scaler = scaler.fit_transform(descriptor_data)
DataFrame_data_scaler = pd.DataFrame(data_scaler)

print("Calculating Pearson and Spearman correlations...")
data_pearson = DataFrame_data_scaler.corr(method='pearson').iloc[:, 0]
data_spearman = DataFrame_data_scaler.corr(method='spearman').iloc[:, 0]

def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

def calculate_distcorr(i):
    return distcorr(data_scaler[:, i], data_scaler[:, 0])

print("Calculating distance correlations in parallel...")
distance_correlation_list = Parallel(n_jobs=-1)(
    delayed(calculate_distcorr)(i) for i in tqdm(range(0, data_scaler.shape[1]), disable=True) 
)

print("Calculating MIC (serial, due to parallelization limitations)...")
mic_correlation_list = []
mine = MINE(alpha=0.6, c=15)
for i in tqdm(range(0, data_scaler.shape[1]), disable=True):
    if i % 500 == 0:
        print(f"Processing MIC for column {i}/{data_scaler.shape[1]}...")
    mine.compute_score(data_scaler[:, i], data_scaler[:, 0])
    mic_correlation_list.append(mine.mic())

def selection_by_threshold(correlation_list, threshold):
    return [1 if abs(corr) > threshold else 0 for corr in correlation_list]

pearson_selection = selection_by_threshold(data_pearson, pearson_threshold)
spearman_selection = selection_by_threshold(data_spearman, spearman_threshold)
distance_selection = selection_by_threshold(distance_correlation_list, distance_threshold)
mic_selection = selection_by_threshold(mic_correlation_list, mic_threshold)

sum_list = np.array(pearson_selection) + np.array(spearman_selection) + np.array(distance_selection) + np.array(mic_selection)

assert len(sum_list) == descriptor_data.shape[1], "sum_list length does not match descriptor_data columns"

descriptor_filter1 = descriptor_data.loc[:, sum_list >= 1]
descriptor_filter2 = descriptor_data.loc[:, sum_list >= 2]
descriptor_filter3 = descriptor_data.loc[:, sum_list >= 3]
descriptor_filter4 = descriptor_data.loc[:, sum_list >= 4]

filter_data1 = pd.concat([y, descriptor_filter1], axis=1)
filter_data2 = pd.concat([y, descriptor_filter2], axis=1)
filter_data3 = pd.concat([y, descriptor_filter3], axis=1)
filter_data4 = pd.concat([y, descriptor_filter4], axis=1)

filter_data4.to_csv(stage_02_2_output_file_path, index=False)

print(f"Number of descriptor columns after correlation filtering: {filter_data4.shape}")
print(f"Correlation coefficient screening completed! >> {stage_02_2_output_file_path}")

print("\n>>>>>>>>>>>>>>>>>>> 03XGBforOpt_descriptors.py >>>>>>>>>>>>>>")
xgb_df = pd.read_csv(stage_02_2_output_file_path)

X_all = xgb_df.drop(xgb_df.columns[0:3], axis=1)
y_all = xgb_df["target"]
scaler = StandardScaler()
X = scaler.fit_transform(X_all)

train_data, test_data = train_test_split(xgb_df, test_size=0.2, random_state=42) 

X_train = train_data.drop(train_data.columns[0:3], axis=1)
y_train = train_data["target"]
X_test = test_data.drop(test_data.columns[0:3], axis=1)
y_test = test_data["target"]

def XGBOpt(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=1,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

pbounds = {
    'n_estimators': (50, 1000),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

optimizer = BayesianOptimization(f=XGBOpt, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)

best_params = optimizer.max['params']
n_estimators = int(best_params['n_estimators'])
max_depth = int(best_params['max_depth'])
learning_rate = best_params['learning_rate']
subsample = best_params['subsample']
colsample_bytree = best_params['colsample_bytree']

print("Optimized parameters:", best_params)

xgb_model = xgb.XGBRegressor(
    n_estimators=n_estimators, 
    max_depth=max_depth, 
    learning_rate=learning_rate, 
    subsample=subsample, 
    colsample_bytree=colsample_bytree, 
    random_state=1,
    n_jobs=-1
)

# Fit the model to get feature importances
xgb_model.fit(X, y_all)

# Get feature importances from the XGBoost model
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

results, MSE, names = [], [], []
best_score = -np.inf

r2_file = open('./data/3-XGB_opt_R2.csv', 'w')
mse_file = open('./data/3-XGB_opt_MSE.csv', 'w')
r2_file.write('Name,descriptor_num,R2,R2_Std\n')
mse_file.write('Name,descriptor_num,MSE,MSE_Std\n')

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) 
i = best_num_features 
selected_features = X_all.columns[indices[:i]]
X_selected = X[:, indices[:i]]

# Perform cross-validation
scores_r2 = cross_val_score(xgb_model, X_selected, y_all, scoring='r2', cv=cv, n_jobs=-1)
scores_mse = cross_val_score(xgb_model, X_selected, y_all, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

# Store results
r2_mean = np.mean(scores_r2)
r2_std = np.std(scores_r2)
mse_mean = -1 * np.mean(scores_mse)
mse_std = np.std(scores_mse)

results.append(scores_r2)
MSE.append(scores_mse)
names.append(i)

r2_file.write(f'R2,{i},{r2_mean:.3f},{r2_std:.3f}\n')
mse_file.write(f'MSE,{i},{mse_mean:.3f},{mse_std:.3f}\n')

df_selected_features = xgb_df[selected_features]

to_save = xgb_df[["ID", "smiles", "target"]].join(df_selected_features)
xgb_file = stage_03_output_file_path
to_save.to_csv(xgb_file, index=False) 
r2_file.close()
mse_file.close()

print(f"Best number of features: {best_num_features}")
print(f"Selected features:\n{selected_features}")
print(f"Machine learning (XGBoost) descriptors filtering complete! >> {xgb_file}")


print("\n>>>>>>>>>>>>>>>>>>> 04smiles-enumeration-id.py >>>>>>>>>>>>>>")

df_04 = pd.read_csv(stage_03_output_file_path)

def normalize_and_generate_smiles(smiles_04, num_variants):
    mol_04 = Chem.MolFromSmiles(smiles_04)
    if not mol_04:
        return None, []  # Return None and empty list if SMILES is invalid
    
    # Normalize SMILES
    normalized_smiles = Chem.MolToSmiles(mol_04, isomericSmiles=True, kekuleSmiles=False)
    
    variants = set()
    # Generate multiple random SMILES by randomizing the atom order
    for _ in range(num_variants):
        random_smiles = Chem.MolToSmiles(mol_04, doRandom=True, kekuleSmiles=False)
        variants.add(random_smiles)
    
    return normalized_smiles, list(variants)

# Iterate through the parameter list
for num_variants in max_smi_num_list:
    print(f"Processing SMILES with {num_variants} random variants per molecule...")
    
    # Initialize an empty list to store the new data
    new_data_04 = []
    
    # Iterate through the dataset to normalize and generate equivalent SMILES for each molecule
    for _, row in tqdm(df_04.iterrows(), total=df_04.shape[0], desc=f"Processing SMILES (variants={num_variants})"):
        smiles_04 = row['smiles']
        molecule_id = row['ID']  # Get the ID
        
        # Normalize and generate SMILES variants
        normalized_smiles, smiles_variants = normalize_and_generate_smiles(smiles_04, num_variants)
        if normalized_smiles is None:
            continue  # Skip invalid SMILES
        
        # Combine the normalized SMILES and the generated random SMILES into one list
        all_smiles = [normalized_smiles] + smiles_variants  # Add normalized SMILES and random ones
        for smile in all_smiles:
            new_data_04.append([molecule_id, smile])  # Store ID and corresponding SMILES
    
    # Convert the new data to a DataFrame
    new_df_04 = pd.DataFrame(new_data_04, columns=['ID', 'smiles'])
    
    # Save the modified data to a new CSV file
    equ_smi_output_file = f'./data/4-smiles_random_order-ID_{num_variants}.csv'
    new_df_04.to_csv(equ_smi_output_file, index=False)
    
    print(f"Processing complete for {num_variants} variants. Generated {len(new_data_04)} records. Results saved to {equ_smi_output_file}.")

    
print("\n>>>>>>>>>>>>>>>>>>> 05-1-gnn-mol_graph.py >>>>>>>>>>>>>>")
print("Only molecular graph structures generated by molecular smiles are included, without rotations and flips !")

def smiles_to_graph(smiles):
    mol_05_1 = Chem.MolFromSmiles(smiles)
    if mol_05_1 is None:
        return None

    node_features = []
    for atom in mol_05_1.GetAtoms():
        atom_feature = [
            atom.GetAtomicNum(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic())
        ]
        node_features.append(atom_feature)

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_features = []
    for bond in mol_05_1.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_features.append([i, j, bond_type])

    if len(edge_features) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long) 
        edge_attr = torch.empty(0, dtype=torch.float) 
    else:
        edge_index = torch.tensor(np.array(edge_features)[:, :2], dtype=torch.long).t().contiguous() 
        edge_attr = torch.tensor(np.array(edge_features)[:, 2:], dtype=torch.float) 

    return node_features, edge_index, edge_attr

df_05_1 = pd.read_csv(stage_03_output_file_path)

graph_data_list = []

for index, row in df_05_1.iterrows():
    smiles = row['smiles']
    graph_data = smiles_to_graph(smiles)
    if graph_data is not None:
        graph_dict = row.to_dict() 
        graph_dict['graph'] = graph_data 
        graph_data_list.append(graph_dict)

with open('./data/5-molecule_graphs-noFilpRotate.pkl', 'wb') as f:
    pickle.dump(graph_data_list, f)
print("The number of molecular graphs (only original, no rotation and flip operations) is ", len(graph_data_list))

print("The mol graph has been successfully saved to './data/5-molecule_graphs-noFilpRotate.pkl'")

print("\n>>>>>>>>>>>>>>>>>>> 05-2-gnn-mol_graph-flip_rotate-parallel.py >>>>>>>>>>>>>>")
print("It contains the molecular graph structure generated by smiles and the rotated and flipped molecular graph !")

def smiles_to_graph(smiles):
    mol_05_2 = Chem.MolFromSmiles(smiles)
    if mol_05_2 is None:
        return None

    node_features = []
    for atom in mol_05_2.GetAtoms():
        atom_feature = [
            atom.GetAtomicNum(), 
            atom.GetTotalNumHs(), 
            int(atom.GetIsAromatic()) 
        ]
        node_features.append(atom_feature)

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_features = []
    for bond in mol_05_2.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_features.append([i, j, bond_type])

    if len(edge_features) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long) 
        edge_attr = torch.empty(0, dtype=torch.float) 
    else:
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
    mol_05_2 = Chem.MolFromSmiles(smiles)
    if mol_05_2 is None:
        return None

    mol_05_2 = Chem.AddHs(mol_05_2)
    start_time = time.time()

    try:
        if AllChem.EmbedMolecule(mol_05_2, maxAttempts=10000) != 0:
            return None 

        if time.time() - start_time > timeout:
            return None

        AllChem.UFFOptimizeMolecule(mol_05_2, maxIters=200)
        conf = mol_05_2.GetConformer()

        for atom_idx in range(mol_05_2.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(atom_idx))
            new_pos = np.dot(transformation, pos)
            conf.SetAtomPosition(atom_idx, new_pos)

        return smiles_to_graph(Chem.MolToSmiles(mol_05_2))
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

df_05_2 = pd.read_csv(stage_03_output_file_path)

results = Parallel(n_jobs=32)(delayed(process_molecule)(row) for _, row in df_05_2.iterrows())

# 将结果展平
graph_data_list = [item for sublist in results if sublist is not None for item in sublist]

with open('./data/5-molecule_graphs-flip_rotate.pkl', 'wb') as f:
    pickle.dump(graph_data_list, f)
print("The number of molecular graphs (original, rotation, and flip operations) is ", len(graph_data_list))
print("The mol graph has been successfully saved to './data/5-molecule_graphs-flip_rotate.pkl'")

print("\n>>>>>>>>>>>>>>>>>>> 06data-enhancement-only_id_smi_graph.py >>>>>>>>>>>>>>")

graph_files = [
    "./data/5-molecule_graphs-flip_rotate.pkl",
    "./data/5-molecule_graphs-noFilpRotate.pkl"
]
smiles_files_template = "./data/4-smiles_random_order-ID_{}.csv"

for graph_file in graph_files:
    with open(graph_file, 'rb') as f:
        molecule_graphs = pickle.load(f)

    print(f"The number of rows in {graph_file}: {len(molecule_graphs)}")

    molecule_graphs_dict = {}
    for entry in molecule_graphs:
        mol_id = entry['ID']
        if mol_id not in molecule_graphs_dict:
            molecule_graphs_dict[mol_id] = []
        molecule_graphs_dict[mol_id].append(entry['graph'])

    for num_variants in max_smi_num_list:
        smiles_file = smiles_files_template.format(num_variants)
        print(f"Processing {graph_file} with {smiles_file}...")

        total_enhanced = 0
        all_data_enhanced_total = []

        smiles_iter = pd.read_csv(smiles_file, chunksize=1000000, on_bad_lines='skip')

        for smiles_chunk in tqdm(smiles_iter, desc=f"Processing {smiles_file}", mininterval=600):
            all_data_enhanced = [] 
            for _, row in smiles_chunk.iterrows():
                smiles_id = row['ID']
                smiles = row['smiles']

                if smiles_id in molecule_graphs_dict:
                    for graph in molecule_graphs_dict[smiles_id]:
                        new_entry = {
                            'ID': smiles_id,
                            'smiles': smiles,
                            'graph': graph
                        }
                        all_data_enhanced.append(new_entry)

            all_data_enhanced_total.extend(all_data_enhanced)
            total_enhanced += len(all_data_enhanced)

        output_file_06 = f"./data/6-all_data_ID-smi-graph-data_enhance-{total_enhanced}.pkl"

        with open(output_file_06, 'wb') as f_out:
            pickle.dump(all_data_enhanced_total, f_out)

        print(f"By combining {graph_file} and {smiles_file}, the number of rows in the data enhancemen dataset is: {total_enhanced}")


