import pandas as pd
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from peft import get_peft_model, LoraConfig
import optuna


model_name = "./BARTSmiles"  
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) 

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target_modules = [
    'model.encoder.layers.11.self_attn.k_proj', 
    'model.encoder.layers.11.self_attn.v_proj', 
    'model.encoder.layers.11.self_attn.q_proj', 
    'model.encoder.layers.11.self_attn.out_proj',
] 

df = pd.read_csv('test-Opt_XGB_descriptor.csv')

smiles_list = df['smiles'].values
target_values = df['log2(AE)'].values.astype(np.float32)

scaler = MinMaxScaler()
target_values_normalized = scaler.fit_transform(target_values.reshape(-1, 1)).flatten()

def smiles_to_tokens(smiles, max_length):
    tokens = tokenizer(smiles, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_length):
        self.smiles_list = smiles_list
        self.targets = targets.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smile = self.smiles_list[idx]
        target = self.targets[idx]
        input_ids, attention_mask = smiles_to_tokens(smile, self.max_length)
        target = torch.tensor(target, dtype=torch.float32)
        return input_ids.clone().detach(), attention_mask.clone().detach(), target.clone().detach()

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    targets = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return input_ids, attention_mask, targets

class BARTSmilesModel(nn.Module):
    def __init__(self, pretrained_model):
        super(BARTSmilesModel, self).__init__()
        self.model = pretrained_model
        self.fc = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        prediction = self.fc(pooled_output)
        return prediction.squeeze()

def create_optimizer(model, lr):
    return optim.AdamW(model.parameters(), lr=lr)

criterion = nn.MSELoss()

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    for input_ids, attention_mask, target in train_loader:
        input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(output.cpu().detach().numpy())
        all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    r2_train = r2_score(all_targets, all_preds)
    mse_train = mean_squared_error(all_targets, all_preds)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(all_targets, all_preds)
    std_dev_train = np.std(all_preds)

    return total_loss / len(train_loader), r2_train, mse_train, rmse_train, mae_train, std_dev_train

def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for input_ids, attention_mask, target in loader:
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    all_preds_denormalized = scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    all_targets_denormalized = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    r2 = r2_score(all_targets_denormalized, all_preds_denormalized)
    mse = mean_squared_error(all_targets_denormalized, all_preds_denormalized)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets_denormalized, all_preds_denormalized)
    std_dev = np.std(all_preds_denormalized)
    return r2, mse, rmse, mae, std_dev

def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    r = trial.suggest_int("r", low=4, high=16, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", low=8, high=32, step=8)
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.3)
    max_length = trial.suggest_categorical("max_length", [1024])
    num_epochs = trial.suggest_int("num_epochs", 20, 120)

    kf = KFold(n_splits=5, shuffle=True, random_state=82)
    r2_scores = []

    for train_idx, val_idx in kf.split(smiles_list):
        train_dataset = SMILESDataset(smiles_list[train_idx], target_values_normalized[train_idx], tokenizer, max_length)
        val_dataset = SMILESDataset(smiles_list[val_idx], target_values_normalized[val_idx], tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        pretrained_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = BARTSmilesModel(pretrained_model)
        lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=dropout_rate, bias="none", task_type="regression", target_modules=target_modules)
        model = get_peft_model(model, lora_config).to(device)
        optimizer = create_optimizer(model, lr)

        for epoch in range(num_epochs):
            train_loss, r2_train, mse_train, rmse_train, mae_train, std_dev_train = train(model, train_loader, optimizer, criterion)
            if (epoch + 1) % 20 == 0:
                r2_val, mse_val, rmse_val, mae_val, std_dev_val = evaluate(model, val_loader)
                print(f"Epoch {epoch + 1}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  <Train> R²: {r2_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, Std Dev: {std_dev_train:.4f}")
                print(f"  <Validation> R²: {r2_val:.4f}, MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, Std Dev: {std_dev_val:.4f}")

        r2 = evaluate(model, val_loader)[0]
        r2_scores.append(r2)

    return np.mean(r2_scores)

def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2) # 60
    print("Best parameters:", study.best_params)
    return study.best_params

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_params = run_optuna()
