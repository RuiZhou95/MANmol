import torch
import torch.nn as nn
from transformers import RobertaModel
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from transformers import RobertaTokenizer
from torch_geometric.nn import global_mean_pool
import torch.distributed as dist
from peft import get_peft_model, LoraConfig
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from matplotlib.ticker import AutoMinorLocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_merge_data(ID_smiles_graph_path, properties_path, chunk_size=100000):
    properties_df = pd.read_csv(properties_path, index_col='ID').drop(columns=['smiles'], errors='ignore')
    merged_data = []

    with open(ID_smiles_graph_path, 'rb') as f:
        while True:
            try:
                all_data = pickle.load(f)
                for item in all_data:
                    if item['ID'] in properties_df.index:
                        properties = properties_df.loc[item['ID']].to_dict()
                        merged_entry = {**item, **properties}
                        merged_data.append(merged_entry)
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
                break

    return merged_data

def load_and_split_data(data):
    smiles = [item['smiles'] for item in data]
    graphs = [item['graph'] for item in data]
    descriptors = [[v for k, v in item.items() if k not in ['ID', 'smiles', 'graph', 'target']] for item in data]
    targets = [item['target'] for item in data]

    smiles_inputs = [tokenizer(smile, return_tensors='pt', padding=True, truncation=True) for smile in smiles]
    
    descriptors = [torch.tensor(desc) for desc in descriptors]

    X_train, X_temp, y_train, y_temp = train_test_split(list(zip(smiles_inputs, graphs, descriptors)), targets, test_size=0.5, random_state=42) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def process_data(X, y, max_len=512):
    """
    Convert data containing SMILES, graph structures, and descriptors into TensorDataset format.
    Args:
        X (list): The X dataset, where each sample contains SMILES, graph structures, and descriptors.
        y (list): The y dataset, containing a value of the target property ('target').
        max_len (int): The maximum length for padding SMILES inputs to ensure all inputs have the same length.

    Returns:
        TensorDataset: A TensorDataset containing input features and the target property.
    """
    smiles_input_ids = []
    smiles_attention_mask = []
    graph_x = []
    graph_edge_index = []
    graph_edge_attr = []
    descriptors = []
    targets = []

    for i in range(len(X)):
        smiles = X[i][0]  # {'input_ids': tensor, 'attention_mask': tensor}
        input_ids = smiles['input_ids']
        attention_mask = smiles['attention_mask']
        
        if input_ids.size(1) < max_len: 
            padding_len = max_len - input_ids.size(1)
            padding = torch.zeros((input_ids.size(0), padding_len), dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding], dim=1)

        if attention_mask.size(1) < max_len:
            padding_len = max_len - attention_mask.size(1)
            padding = torch.zeros((attention_mask.size(0), padding_len), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, padding], dim=1)

        smiles_input_ids.append(input_ids)
        smiles_attention_mask.append(attention_mask)

        graph = X[i][1]  # (tensor, tensor, tensor)
        graph_x.append(graph[0])
        graph_edge_index.append(graph[1])
        graph_edge_attr.append(graph[2])

        descriptors.append(X[i][2]) 

        targets.append(y[i])

    smiles_input_ids = torch.stack(smiles_input_ids)
    smiles_attention_mask = torch.stack(smiles_attention_mask)
    descriptors = torch.stack(descriptors)
    targets = torch.tensor(targets) 

    graph_x = pad_sequence(graph_x, batch_first=True, padding_value=0)
    
    max_len = max([edge_index.size(1) for edge_index in graph_edge_index])

    padded_graph_edge_index = [
        torch.cat([edge_index, torch.zeros(2, max_len - edge_index.size(1), dtype=torch.long)], dim=1)
        if edge_index.size(1) < max_len else edge_index
        for edge_index in graph_edge_index
    ]
    
    graph_edge_index = pad_sequence(padded_graph_edge_index, batch_first=True, padding_value=0)

    padded_graph_edge_attr = []
    for edge_attr in graph_edge_attr:
        if edge_attr.size(0) == 0: 
            padded_graph_edge_attr.append(torch.zeros((0, 1), dtype=edge_attr.dtype))  # 1表示特征维度
        else:
            padded_graph_edge_attr.append(edge_attr)

    graph_edge_attr = pad_sequence(padded_graph_edge_attr, batch_first=True, padding_value=0)

    dataset = TensorDataset(smiles_input_ids, smiles_attention_mask, graph_x, graph_edge_index, graph_edge_attr, descriptors, targets)
    
    return dataset


class GNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(GNN, self).__init__()

        self.fc_input = nn.Linear(3, hidden_size)
        self.convs = nn.ModuleList([
            GCNConv(in_channels=hidden_size if i == 0 else hidden_size, out_channels=hidden_size)
            for i in range(num_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.fc_input(x)

        if edge_attr is not None:
            edge_attr_list = []
            for i in range(edge_attr.size(0)):
                edge_attr_list.append(edge_attr[i].view(-1, edge_attr.size(-1)))
            edge_attr = torch.cat(edge_attr_list, dim=0)
            # print('Processed edge_attr.shape:', edge_attr.shape)
        else:
            print("Warning: No edge_attr provided.")

        edge_index_list = []
        for i in range(edge_index.size(0)): 
            edge_index_list.append(edge_index[i])
        edge_index = torch.cat(edge_index_list, dim=1) 

        x_list = []
        for i in range(x.size(0)): 
            x_list.append(x[i])
        x = torch.cat(x_list, dim=0)

        for conv, norm in zip(self.convs, self.norms):
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr) 
            else:
                x = conv(x, edge_index)

            x = norm(x)
            x = torch.relu(x)

        x = self.fc(x)
        x = self.dropout(x)

        return x

class MultiModalAttentionModelWithMLP(nn.Module):
    def __init__(self, chemberta_model_name, descriptor_size, gnn_hidden_size):
        super(MultiModalAttentionModelWithMLP, self).__init__()
        
        self.chemberta = RobertaModel.from_pretrained(chemberta_model_name, ignore_mismatched_sizes=True)
        
        self.fc_descriptor = nn.Linear(descriptor_size, 256)
        
        self.gnn = GNN(gnn_hidden_size, num_layers=3)  
        
        self.smiles_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.descriptor_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        self.graph_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        
        self.attention_smiles = nn.MultiheadAttention(embed_dim=384, num_heads=8)
        self.attention_descriptor = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.attention_graph = nn.MultiheadAttention(embed_dim=gnn_hidden_size, num_heads=8)
        
        self.mlp_smiles = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.mlp_descriptor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mlp_graph = nn.Sequential(
            nn.Linear(gnn_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.fc_combined = nn.Linear(128, 1)

        self.lora_config = LoraConfig(
            r=8,  
            lora_alpha=16,  
            lora_dropout=0.1,
            bias="none",
            task_type="regression",
            target_modules=["attention.self.query", "attention.self.key", "attention.self.value"]
        )

        self.chemberta = get_peft_model(self.chemberta, self.lora_config)

        self.lora_config_gnn = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="regression",
            target_modules=["fc_input", "fc"]
        )

        self.gnn = get_peft_model(self.gnn, self.lora_config_gnn)       

    def forward(self, smiles_input, descriptors, graph_data):
        # Ensure input_ids and attention_mask are 2D (batch_size, seq_length)
        smiles_input['input_ids'] = smiles_input['input_ids'].view(-1, smiles_input['input_ids'].size(-1))  # Reshape to 2D
        smiles_input['attention_mask'] = smiles_input['attention_mask'].view(-1, smiles_input['attention_mask'].size(-1))  # Reshape to 2D

        smiles_output = self.chemberta(**smiles_input).last_hidden_state   
        smiles_attn_output, _ = self.attention_smiles(smiles_output, smiles_output, smiles_output)
        smiles_mlp_output = self.mlp_smiles(smiles_attn_output.mean(dim=1))

        descriptors_tensor = descriptors.clone().detach().float()      
        descriptor_output = self.fc_descriptor(descriptors_tensor)

        descriptor_output = self.fc_descriptor(descriptors_tensor)  # Shape: (batch_size, 256)
        descriptor_output = descriptor_output.unsqueeze(1)  # Reshape to (batch_size, 1, 256) for attention
        
        descriptor_attn_output, _ = self.attention_descriptor(descriptor_output, descriptor_output, descriptor_output)
        descriptor_mlp_output = self.mlp_descriptor(descriptor_attn_output.mean(dim=1))

        if isinstance(graph_data, Data):
            batch = graph_data
        else:
            try:
                batch = Batch.from_data_list([graph_data])
            except Exception as e:
                raise ValueError(f"Invalid graph_data format. Error: {e}")

        x = batch.x.clone().detach().float() 
        edge_index = batch.edge_index.clone().detach().long() 
        edge_attr = batch.edge_attr.clone().detach().float() if batch.edge_attr is not None else None 

        graph_output = self.gnn(batch)

        graph_output = global_mean_pool(graph_output, batch=batch.batch)
        graph_mlp_output = self.mlp_graph(graph_output)

        total_weight = torch.cat([self.smiles_weight, self.descriptor_weight, self.graph_weight], dim=0)
        adaptive_weights = torch.softmax(total_weight, dim=0)

        combined_output = (adaptive_weights[0] * smiles_mlp_output + 
                           adaptive_weights[1] * descriptor_mlp_output + 
                           adaptive_weights[2] * graph_mlp_output)

        predictions = self.fc_combined(combined_output)       
        predictions = predictions.squeeze(0)      

        return predictions

def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs, save_path, log_file, target_names):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model.to(device)

    if dist.get_rank() == 0:
        with open(log_file, 'w') as f:
            f.write('Epoch,Train_Loss,Val_Loss,Test_Loss,Train_MSE,Test_MSE,Valid_MSE,Train_R2,Test_R2,Valid_R2,Train_RMSE,Test_RMSE,Valid_RMSE,Train_MAE,Test_MAE,Valid_MAE\n')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mse_list = []  
        train_r2_list = []
        train_mae_list = []
        train_rmse_list = []

        predictions_train, ground_truth_train = [], []

        for batch in train_loader:
            smiles_input_ids, smiles_attention_mask, graph_x, graph_edge_index, graph_edge_attr, descriptors, targets = batch

            smiles_input = {
                'input_ids': smiles_input_ids.to(device),
                'attention_mask': smiles_attention_mask.to(device)
            }
            descriptors = descriptors.to(device)

            graph_data = Data(
                x=graph_x.to(device),
                edge_index=graph_edge_index.to(device),
                edge_attr=graph_edge_attr.to(device) if graph_edge_attr is not None else None
            )

            optimizer.zero_grad()
            output = model(smiles_input, descriptors, graph_data)  
            targets = targets.to(device)
            output = output.squeeze()  
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()            

            target_value = targets.cpu().numpy()
            output_value = output.cpu().detach().numpy()

            if len(target_value) > 1:
                mse = mean_squared_error(target_value, output_value)
                r2 = r2_score(target_value, output_value)
                mae = mean_absolute_error(target_value, output_value)
                rmse = np.sqrt(mse)
            else:
                mse = np.nan
                r2 = np.nan
                mae = np.nan
                rmse = np.nan

            train_mse_list.append(mse)
            train_r2_list.append(r2)
            train_mae_list.append(mae)
            train_rmse_list.append(rmse)

            predictions_train.append(output.cpu().detach().numpy())
            ground_truth_train.append(targets.cpu().numpy())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_mse_list = []  
        val_r2_list = []   
        val_mae_list = []
        val_rmse_list = []

        predictions_val, ground_truth_val = [], []

        with torch.no_grad():
            for batch in val_loader:
                smiles_input_ids, smiles_attention_mask, graph_x, graph_edge_index, graph_edge_attr, descriptors, targets = batch

                smiles_input = {'input_ids': smiles_input_ids.to(device), 'attention_mask': smiles_attention_mask.to(device)}
                descriptors = descriptors.to(device)

                graph_data = Data(
                    x=graph_x.to(device),
                    edge_index=graph_edge_index.to(device),
                    edge_attr=graph_edge_attr.to(device) if graph_edge_attr is not None else None
                )

                output = model(smiles_input, descriptors, graph_data)
                targets = targets.to(device)
                output = output.squeeze()  
                loss = criterion(output, targets)
                val_loss += loss.item()
                
                predictions_val.append(output.cpu().detach().numpy())
                ground_truth_val.append(targets.cpu().numpy())                

                target_value = targets.cpu().numpy()
                output_value = output.cpu().detach().numpy()

                if len(target_value) > 1:
                    mse = mean_squared_error(target_value, output_value)
                    r2 = r2_score(target_value, output_value)
                    mae = mean_absolute_error(target_value, output_value)
                    rmse = np.sqrt(mse)
                else:
                    mse = np.nan
                    r2 = np.nan
                    mae = np.nan
                    rmse = np.nan

                val_mse_list.append(mse)
                val_r2_list.append(r2)
                val_mae_list.append(mae)
                val_rmse_list.append(rmse)

        val_loss /= len(val_loader)            

        # Test
        model.eval()
        test_loss = 0
        test_mse_list = []  
        test_r2_list = []   
        test_mae_list = []
        test_rmse_list = []

        predictions_test, ground_truth_test = [], []

        with torch.no_grad():
            for batch in test_loader:
                smiles_input_ids, smiles_attention_mask, graph_x, graph_edge_index, graph_edge_attr, descriptors, targets = batch

                smiles_input = {'input_ids': smiles_input_ids.to(device), 'attention_mask': smiles_attention_mask.to(device)}
                descriptors = descriptors.to(device)

                graph_data = Data(
                    x=graph_x.to(device),
                    edge_index=graph_edge_index.to(device),
                    edge_attr=graph_edge_attr.to(device) if graph_edge_attr is not None else None
                )

                output = model(smiles_input, descriptors, graph_data)
                targets = targets.to(device)
                output = output.squeeze()  
                loss = criterion(output, targets)
                test_loss += loss.item()

                predictions_test.append(output.cpu().detach().numpy())
                ground_truth_test.append(targets.cpu().numpy())

                target_value = targets.cpu().numpy()
                output_value = output.cpu().detach().numpy()

                if len(target_value) > 1:
                    mse = mean_squared_error(target_value, output_value)
                    r2 = r2_score(target_value, output_value)
                    mae = mean_absolute_error(target_value, output_value)
                    rmse = np.sqrt(mse)
                else:
                    mse = np.nan
                    r2 = np.nan
                    mae = np.nan
                    rmse = np.nan

                test_mse_list.append(mse)
                test_r2_list.append(r2)
                test_mae_list.append(mae)
                test_rmse_list.append(rmse)

        test_loss /= len(test_loader)
         
        if dist.get_rank() == 0:
            log_data = {
                'Epoch': [epoch + 1],
                'Train_Loss': [train_loss],
                'Val_Loss': [val_loss],
                'Test_Loss': [test_loss],
                'Train_MSE': [np.mean(train_mse_list)],
                'Test_MSE': [np.mean(test_mse_list)],
                'Valid_MSE': [np.mean(val_mse_list)],
                'Train_R2': [np.mean(train_r2_list)],
                'Test_R2': [np.mean(test_r2_list)],
                'Valid_R2': [np.mean(val_r2_list)],
                'Train_RMSE': [np.mean(train_rmse_list)],
                'Test_RMSE': [np.mean(test_rmse_list)],
                'Valid_RMSE': [np.mean(val_rmse_list)],
                'Train_MAE': [np.mean(train_mae_list)],
                'Test_MAE': [np.mean(test_mae_list)],
                'Valid_MAE': [np.mean(val_mae_list)],
            }

            log_df = pd.DataFrame(log_data)
            cols_to_round = log_df.columns.difference(['Epoch'])
            log_df[cols_to_round] = log_df[cols_to_round].round(4)
            log_df.to_csv(log_file, mode='a', header=False, index=False)

            print(f'\nEpoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        predictions_train = np.concatenate(predictions_train, axis=0)
        ground_truth_train = np.concatenate(ground_truth_train, axis=0)    

        predictions_val = np.concatenate(predictions_val, axis=0)
        ground_truth_val = np.concatenate(ground_truth_val, axis=0)    

        predictions_test = np.concatenate(predictions_test, axis=0)
        ground_truth_test = np.concatenate(ground_truth_test, axis=0)

        if epoch == epochs - 1:
            try:
                plot_diagonal(predictions_train, ground_truth_train, predictions_val, ground_truth_val, predictions_test, ground_truth_test, target_names, fig_output_dir)
            except Exception as e:
                print(f'Error in plotting results: {e}')

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), save_path)
        
    return test_loss, predictions_train, ground_truth_train, predictions_test, ground_truth_test, predictions_val, ground_truth_val, val_r2_list, val_mse_list


         
def plot_diagonal(predictions_train, ground_truth_train, predictions_val, ground_truth_val, predictions_test, ground_truth_test, target_names, fig_output_dir):
    """
    Draw a diagonal plot and save it
    :param predictions_train: Predicted values of the model on the training set
    :param ground_truth_train: True labels of the training set
    :param predictions_val: Predicted values of the model on the validation set
    :param ground_truth_val: True labels of the validation set
    :param predictions_test: Predicted values of the model on the test set
    :param ground_truth_test: True labels of the test set
    :param target_names: Names of the target properties
    :param output_dir: Output directory
    """
    
    def to_1d_array(data):
        if isinstance(data, (list, np.ndarray)):
            return np.asarray(data).flatten()
        raise ValueError(f"The input data format is incorrect. It needs to be a list or array: {data}")

    predictions_train = to_1d_array(predictions_train)
    ground_truth_train = to_1d_array(ground_truth_train)
    predictions_val = to_1d_array(predictions_val)
    ground_truth_val = to_1d_array(ground_truth_val)
    predictions_test = to_1d_array(predictions_test)
    ground_truth_test = to_1d_array(ground_truth_test)

    all_preds = [predictions_train, predictions_val, predictions_test]
    all_gts = [ground_truth_train, ground_truth_val, ground_truth_test]
    labels = ['Train', 'Validation', 'Test']

    inch2cm = 1 / 2.54
    plt.figure(figsize=(24 * inch2cm, 17 * inch2cm))
    plt.rc('font', family='Times New Roman', weight='normal')
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

    plt.figure(figsize=(8, 8))
    
    for j in range(3):
        preds = all_preds[j]
        gts = all_gts[j]

        marker = 'o' if j == 0 else ('o' if j == 1 else 'o')
        size = 40 if j == 0 else (50 if j == 1 else 60)
        color = '#252a34' if j == 0 else ('#08d9d6' if j == 1 else '#ff2e63') 
        zorder = j - 3 
        
        plt.scatter(gts, preds, alpha=0.2, label=f"{labels[j]}", marker=marker, s=size, c=color, zorder=zorder)

    all_min = min(np.min([gt.min() for gt in all_gts]), np.min([pred.min() for pred in all_preds]))
    all_max = max(np.max([gt.max() for gt in all_gts]), np.max([pred.max() for pred in all_preds]))
    plt.plot([all_min, all_max], [all_min, all_max], color='#C0C0C0', linestyle='--', linewidth=3.5, alpha=0.8, zorder=1)

    plt.xlabel(f'Predicted {target_names}', fontsize=22)
    plt.ylabel(f'True {target_names}', fontsize=22)
    plt.xlim([all_min, all_max])
    plt.ylim([all_min, all_max])

    axes = plt.gca()
    axes.minorticks_on()

    axes.tick_params(axis="x", which="major", direction="out", width=1.5, length=5)
    axes.tick_params(axis="y", which="major", direction="out", width=1.5, length=5)
    axes.tick_params(axis="x", which="minor", direction="out", width=1.5, length=3)
    axes.tick_params(axis="y", which="minor", direction="out", width=1.5, length=3)

    axes.xaxis.set_minor_locator(AutoMinorLocator(2))
    axes.yaxis.set_minor_locator(AutoMinorLocator(2))

    TK = plt.gca()
    bwith = 1.5
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    TK.spines['bottom'].set_color('k')
    TK.spines['left'].set_color('k')
    TK.spines['top'].set_color('k')
    TK.spines['right'].set_color('k')
    plt.xticks(fontname='Times New Roman', fontsize=18)
    plt.yticks(fontname='Times New Roman', fontsize=18)

    plt.legend(prop={'family': 'Times New Roman', 'size': 22})
    plt.tight_layout()
    
    plt.savefig(f"{fig_output_dir}{target_names}_diagonal_plot.png", dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def init_distributed_mode():
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(dist.get_rank())
    else:
        raise RuntimeError("CUDA is not available")

if __name__ == "__main__":
    init_distributed_mode()

    # ID_smiles_graph_path = '/public/home/zhour/codeLab/EPAA-ml/LLM/data/fine_tuning_data/all_data_ID-smiles-graph-test.pkl'
    # properties_path = '/public/home/zhour/codeLab/EPAA-ml/LLM/data/all_porperty_AE_data-test.csv'
    # ID_smiles_graph_path = './data/6-all_data_ID-smiles-graph.pkl' # -1843331   -14.5M
    # properties_path = './data/3-Opt_XGB_descriptor.csv' 
    ID_smiles_graph_path = './data/6-all_data_ID-smi-graph-data_enhance-2833413.pkl' # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    properties_path = './data/3-Opt_XGB_descriptor.csv' 
    
    model_save_path = './LubLLM-MTR/LubBERT.model'
    log_file = './figure/MTR-R2-MSE-MAE.csv'
    fig_output_dir='./figure/' 
    # target_names = ['LogP', 'TPSA', 'SA_Score', 'log2(AE)']
    target_names = 'target'
    
    merged_data = load_and_merge_data(ID_smiles_graph_path, properties_path)
    
    tokenizer = RobertaTokenizer.from_pretrained('/public/home/zhour/codeLab/EPAA-ml/LLM/MLM-model/MLM-65M/LubLLM-MLM')

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(merged_data)

    train_dataset = process_data(X_train, y_train)
    val_dataset = process_data(X_val, y_val)    
    test_dataset = process_data(X_test, y_test)    
    
    batch_size = 128
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    model = MultiModalAttentionModelWithMLP(chemberta_model_name='/public/home/zhour/codeLab/EPAA-ml/LLM/MLM-model/MLM-65M/LubLLM-MLM', descriptor_size=33, gnn_hidden_size=64)

    if dist.get_rank() == 0:
        train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=1000, save_path=model_save_path, log_file=log_file, target_names=target_names)
        
    print('>>>>>>>>>>>>>>> Everything you wanted! >>>>>>>>>>>>>>>>>>>>>')