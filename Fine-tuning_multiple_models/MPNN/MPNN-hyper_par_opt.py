import pandas as pd
import torch
import numpy as np
from torch import nn
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
import optuna
from torch_geometric.nn import global_mean_pool, MessagePassing
import torch.nn.functional as F

df = pd.read_csv('Opt_XGB_descriptor-nohighMW.csv')

smiles_list = df['smiles'].values
target_values = df['log2(AE)'].values

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: SMILES {smiles} failed to convert to a molecule.")
        return None
    
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    bond_indices = []
    bond_types = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_indices.append([i, j])
        bond_types.append(bond.GetBondTypeAsDouble())
    
    edge_index = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    edge_attr = torch.tensor(bond_types, dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

graphs = []
for smile in smiles_list:
    graph = smiles_to_graph(smile)
    if graph is not None:
        graphs.append(graph)

target_dict = {i: target_values[i] for i in range(len(target_values))}

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='mean') 
        self.message_fc = nn.Linear(in_channels + 1, out_channels) 
        self.update_fc = nn.Linear(in_channels + out_channels, out_channels)  

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return F.relu(self.message_fc(torch.cat([x_j, edge_attr], dim=-1)))

    def update(self, aggr_out, x):
        return F.relu(self.update_fc(torch.cat([x, aggr_out], dim=-1)))

class MPNNModel(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, dropout_rate=0.3):
        super(MPNNModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(MPNNLayer(1, hidden_dim)) 
        for _ in range(num_layers - 1):
            self.convs.append(MPNNLayer(hidden_dim, hidden_dim))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.dropout(x)
        
        x = global_mean_pool(x, data.batch)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds_train = []
    all_targets_train = []
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        all_preds_train.append(output.squeeze().cpu().detach().numpy())
        all_targets_train.append(target.cpu().numpy())
    
    all_preds_train = np.concatenate(all_preds_train)
    all_targets_train = np.concatenate(all_targets_train)
    
    r2_train = r2_score(all_targets_train, all_preds_train)
    mse_train = mean_squared_error(all_targets_train, all_preds_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(all_targets_train, all_preds_train)
    std_dev_train = np.std(all_preds_train)
    
    return total_loss / len(train_loader), r2_train, mse_train, rmse_train, mae_train, std_dev_train

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.append(output.squeeze().cpu().detach().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    r2 = r2_score(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    std_dev = np.std(all_preds)
    
    return r2, mse, rmse, mae, std_dev

def objective(trial):
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.25, 0.5)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    
    model = MPNNModel(num_layers=num_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs)):
        print(f"Fold {fold+1}")
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        
        train_data = [(train_graphs[i], torch.tensor(target_dict[train_idx[i]], dtype=torch.float)) for i in range(len(train_idx))]
        val_data = [(val_graphs[i], torch.tensor(target_dict[val_idx[i]], dtype=torch.float)) for i in range(len(val_idx))]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        for epoch in range(500):
            train_loss, r2_train, mse_train, rmse_train, mae_train, std_dev_train = train(model, train_loader, optimizer, criterion, device)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  <Train> R²: {r2_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, Std Dev: {std_dev_train:.4f}')
                r2_val, mse_val, rmse_val, mae_val, std_dev_val = test(model, val_loader, device)
                print(f'  <Validation> R²: {r2_val:.4f}, MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, Std Dev: {std_dev_val:.4f}')
                val_r2_scores.append(r2_val)
    
    return np.mean(val_r2_scores)

def optimize():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=60)
    print(f"Best hyperparameters: {study.best_params}")
    return study.best_params

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_params = optimize()
