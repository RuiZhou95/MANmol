import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone() 
        }

df = pd.read_csv("OCSmi.csv")
smiles_list = df['smiles'].tolist()

tokenizer = RobertaTokenizer.from_pretrained('./local_ChemBERTa-77M-MLM')
model = RobertaForMaskedLM.from_pretrained('./local_ChemBERTa-77M-MLM')

dataset = SmilesDataset(smiles_list, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./triberta",
    overwrite_output_dir=True,
    num_train_epochs=100, 
    per_device_train_batch_size=64, 
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./MANmol-65M-MLM")
tokenizer.save_pretrained("./MANmol-65M-MLM")
