import pandas as pd
import torch
import numpy as np
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import torch.optim as optim
from peft import get_peft_model, LoraConfig
import matplotlib.pyplot as plt

model_name = "./ChemBERTa/ChemBERTa-77M-MTR"  
tokenizer = RobertaTokenizer.from_pretrained(model_name)  
target_modules = ["fc"]  

df = pd.read_csv('Opt_XGB_descriptor-nohighMW.csv')
smiles_list = df['smiles'].values
target_values = df['log2(AE)'].values.astype(np.float32)

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
        return input_ids.clone().detach(), attention_mask.clone().detach(), torch.tensor(target, dtype=torch.float32)

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    targets = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return input_ids, attention_mask, targets

class ChemBERTaModel(nn.Module):
    def __init__(self, pretrained_model):
        super(ChemBERTaModel, self).__init__()
        self.bert = pretrained_model
        self.fc = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
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

def evaluate(model, loader, device):
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
    plt.xlabel('ChemBERTa calculated log2(AE) [kcal/mol]', font1)
    plt.ylabel('ChemBERTa predicted log2(AE) [kcal/mol]', font1)
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
    plt.savefig("./ChemBERTa/figure/ChemBERTa-77M.png", dpi=600)

def train_final_model():
    batch_size = 32
    learning_rate = 9.967-5
    lora_r = 13
    lora_alpha = 13
    dropout_rate = 0.157
    max_length = 128
    num_epochs = 156
    
    patience = 20
    min_delta = 1e-8
    best_val_score = -np.inf
    patience_counter = 0

    dataset = SMILESDataset(smiles_list, target_values, tokenizer, max_length)
    train_idx, val_idx = next(KFold(n_splits=5, shuffle=True, random_state=17242).split(dataset)) # 《《《《《《《《《《《《
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = ChemBERTaModel(RobertaModel.from_pretrained(model_name))
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout_rate, task_type="regression", target_modules=target_modules)
    model = get_peft_model(model, lora_config).to(device)
    optimizer = create_optimizer(model, learning_rate)

    for epoch in range(num_epochs):
        train_loss, r2_train, mse_train, rmse_train, mae_train, std_dev_train = train(model, train_loader, optimizer, criterion)
        r2_val, mse_val, rmse_val, mae_val, std_dev_val = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  <Train> R²: {r2_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, Std Dev: {std_dev_train:.4f}")
        print(f"  <Validation> R²: {r2_val:.4f}, MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, Std Dev: {std_dev_val:.4f}")

        if r2_val - best_val_score > min_delta:
            best_val_score = r2_val
            patience_counter = 0
            torch.save(model.state_dict(), "./ChemBERTa/pretrained_model/ChemBERTa-77M-AE_best.model")
            print("Best model saved.")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load("./ChemBERTa/pretrained_model/ChemBERTa-77M-AE_best.model"))
    r2_train_final, mse_train_final, rmse_train_final, mae_train_final, std_dev_train_final = evaluate(model, train_loader, device)
    print(f"Final Model <train> - R²: {r2_train_final:.4f}, MSE: {mse_train_final:.4f}, RMSE: {rmse_train_final:.4f}, MAE: {mae_train_final:.4f}, Std Dev: {std_dev_train_final:.4f}")

    test_loader = DataLoader(torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset]), batch_size=batch_size, shuffle=False)
    r2_final, mse_final, rmse_final, mae_final, std_dev_final = evaluate(model, test_loader, device)
    print(f"Final Model <test> - R²: {r2_final:.4f}, MSE: {mse_final:.4f}, RMSE: {rmse_final:.4f}, MAE: {mae_final:.4f}, Std Dev: {std_dev_final:.4f}")

    plot_diagonal_validation(
        y_train=np.concatenate([target.cpu().numpy() for _, _, target in train_loader]),
        y_train_pred=np.concatenate([model(input_ids.to(device), attention_mask.to(device)).cpu().detach().numpy() for input_ids, attention_mask, _ in train_loader]),
        y_test=np.concatenate([target.cpu().numpy() for _, _, target in test_loader]),
        y_test_pred=np.concatenate([model(input_ids.to(device), attention_mask.to(device)).cpu().detach().numpy() for input_ids, attention_mask, _ in test_loader])
    )

    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_final_model()
