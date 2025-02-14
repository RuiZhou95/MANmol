import pandas as pd
import torch
import numpy as np
from torch import nn
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

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

graphs = [smiles_to_graph(smile) for smile in smiles_list if smiles_to_graph(smile) is not None]
target_dict = {i: target_values[i] for i in range(len(target_values))}

class GCNModel(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, dropout_rate=0.3):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(1, hidden_dim)) 
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = torch.relu(self.fc1(x))
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
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        all_preds_train.append(output.cpu().detach().numpy())
        all_targets_train.append(target.cpu().numpy())

    all_preds_train = np.concatenate(all_preds_train)
    all_targets_train = np.concatenate(all_targets_train)

    r2_train = r2_score(all_targets_train, all_preds_train)
    mse_train = mean_squared_error(all_targets_train, all_preds_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(all_targets_train, all_preds_train)
    std_dev_train = np.std(all_preds_train)

    print(f"  Train - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, Std Dev: {std_dev_train:.4f}")

    return total_loss / len(train_loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    r2 = r2_score(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    std_dev = np.std(all_preds)

    return r2, mse, rmse, mae, std_dev

def plot_diagonal_validation(y_train, y_train_pred, y_test, y_test_pred):
    plt.rc('font', family='Times New Roman', weight='normal')
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    plt.figure(figsize=(6, 5.5))

    plt.plot(y_train, y_train_pred, color='#C0C0C0', marker='o', linestyle='', markersize=3, markerfacecolor='#80C149', alpha=1, markeredgewidth=0.3)
    plt.plot(y_test, y_test_pred, color='#C0C0C0', marker='o', linestyle='', markersize=3, markerfacecolor='b', alpha=1, markeredgewidth=0.3)

    plt.legend(labels=["Train data", "Test data"], loc="lower right", fontsize=18, frameon=False)
    plt.xlabel('MD calculated log2(AE) [kcal/mol]', font1)
    plt.ylabel('GCN predicted log2(AE) [kcal/mol]', font1)
    plt.xlim((4.5, 9.5))
    plt.ylim((4.5, 9.5))
    plt.plot([4.5, 9.5], [4.5, 9.5], color='k', linewidth=1.5, linestyle='--')

    plt.xticks(np.arange(4.5, 9.5, 1.0), size=16)
    plt.yticks(np.arange(4.5, 9.5, 1.0), size=16)
    plt.tick_params(width=1.5)

    TK = plt.gca()
    bwith = 1.5
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    plt.savefig('./figure/GCN.png', dpi=600)

def train_final_model():
    batch_size = 16
    learning_rate = 0.00014
    hidden_dim = 32
    dropout_rate = 0.33218
    num_layers = 2

    patience = 100
    min_delta = 1e-14
    best_val_score = -np.inf
    patience_counter = 0

    model = GCNModel(num_layers=num_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    kfold = KFold(n_splits=5, shuffle=True, random_state=1890) # 《《《《《《《《《《《《《《《
    for train_idx, val_idx in kfold.split(graphs):
        break

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    train_data = [(train_graphs[i], torch.tensor(target_dict[train_idx[i]], dtype=torch.float)) for i in range(len(train_idx))]
    val_data = [(val_graphs[i], torch.tensor(target_dict[val_idx[i]], dtype=torch.float)) for i in range(len(val_idx))]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for epoch in range(500): 
        train_loss = train(model, train_loader, optimizer, criterion, device)
        r2_val, mse_val, rmse_val, mae_val, std_dev_val = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation - R²: {r2_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, Std Dev: {std_dev_val:.4f}")

        if r2_val - best_val_score > min_delta:
            best_val_score = r2_val
            patience_counter = 0
            torch.save(model.state_dict(), "./pretrained_model/GCN_best.model")
            print("Best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load("./pretrained_model/GCN_best.model"))
    print("Best model loaded for final evaluation.")

    r2_train_final, mse_train_final, rmse_train_final, mae_train_final, std_dev_train_final = evaluate(model, train_loader, device)
    print(f"Final Model <train> - R²: {r2_train_final:.4f}, RMSE: {rmse_train_final:.4f}, MAE: {mae_train_final:.4f}, Std Dev: {std_dev_train_final:.4f}")

    test_loader = DataLoader(train_data + val_data, batch_size=batch_size, shuffle=False)
    r2_final, mse_final, rmse_final, mae_final, std_dev_final = evaluate(model, test_loader, device)
    print(f"Final Model <test> - R²: {r2_final:.4f}, RMSE: {rmse_final:.4f}, MAE: {mae_final:.4f}, Std Dev: {std_dev_final:.4f}")

    plot_diagonal_validation(
        y_train=np.concatenate([y for _, y in train_loader]),
        y_train_pred=np.concatenate([model(data.to(device)).cpu().detach().numpy() for data, _ in train_loader]),
        y_test=np.concatenate([y for _, y in test_loader]),
        y_test_pred=np.concatenate([model(data.to(device)).cpu().detach().numpy() for data, _ in test_loader])
    )

    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model = train_final_model()
