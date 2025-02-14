import pandas as pd
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import torch.optim as optim
from peft import get_peft_model, LoraConfig

df = pd.read_csv('test-Opt_XGB_descriptor.csv')

smiles_list = df['smiles'].values
target_values = df['log2(AE)'].values

model_name = './BARTSmiles' 
tokenizer = AutoTokenizer.from_pretrained("./BARTSmiles/", add_prefix_space=True)
model = AutoModel.from_pretrained('./BARTSmiles')

print('*******************************************')
for name, module in model.named_modules():
    print(name) 

def smiles_to_tokens(smiles):
    tokens = tokenizer(smiles, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)    

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, targets, tokenizer):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smile = self.smiles_list[idx]
        target = self.targets[idx]
        input_ids, attention_mask = smiles_to_tokens(smile)
        assert input_ids.shape == attention_mask.shape, f"Shape mismatch: {input_ids.shape} vs {attention_mask.shape}"
        
        target = torch.from_numpy(np.array([target])) 
        return input_ids.clone().detach(), attention_mask.clone().detach(), target.clone().detach()

X_train, X_test, y_train, y_test = train_test_split(smiles_list, target_values, test_size=0.2, random_state=42)
train_dataset = SMILESDataset(X_train, y_train, tokenizer)
test_dataset = SMILESDataset(X_test, y_test, tokenizer)

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    targets = torch.tensor([item[2] for item in batch])
    return input_ids, attention_mask, targets

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

class BARTSmilesModel(nn.Module):
    def __init__(self, pretrained_model):
        super(BARTSmilesModel, self).__init__()
        self.model = pretrained_model
        self.fc = nn.Linear(768, 1) 

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        prediction = self.fc(pooled_output)
        return prediction.squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BARTSmilesModel(model).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.MSELoss()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="regression", 
    target_modules = [
        'encoder.layers.11.self_attn.q_proj',  
        'encoder.layers.11.self_attn.k_proj',
        'encoder.layers.11.self_attn.v_proj',
    ]
)

model = get_peft_model(model, lora_config)

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds_train = []
    all_targets_train = []

    for input_ids, attention_mask, target in train_loader:
        input_ids = input_ids.to(device).long()  # 转换为 LongTensor
        attention_mask = attention_mask.to(device).float()
        target = target.to(device).float()

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
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

    return total_loss / len(train_loader), r2_train, mse_train, rmse_train, mae_train

def test(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for input_ids, attention_mask, target in test_loader:
            input_ids, attention_mask, target = input_ids.to(device).long(), attention_mask.to(device).float(), target.to(device).float()
            output = model(input_ids, attention_mask)
            all_preds.append(output.cpu().detach().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    r2 = r2_score(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)

    return r2, mse, rmse, mae

model_save_path = './pretrained_model/BARTSmiles_AE.model'

for epoch in range(1000):
    train_loss, r2_train, mse_train, rmse_train, mae_train = train(model, train_loader, optimizer, criterion)
    if (epoch + 1) % 10 == 0:
        r2_test, mse_test, rmse_test, mae_test = test(model, test_loader)

        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  <Train> R²: {r2_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}')
        print(f'  <Test> R²: {r2_test:.4f}, MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}')

    torch.save(model.state_dict(), model_save_path)
