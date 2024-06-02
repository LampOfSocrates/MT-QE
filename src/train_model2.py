from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.utils.data import random_split
from model2 import TranslationQualityModel
import pytorch_lightning as pl
import torch 

class TranslationDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'src': row['src'],
            'mt': row['mt'],
            'ref': row['ref'],
            'score': torch.tensor(row['score'], dtype=torch.float)
        }

file_path = 'data/2022-da.csv'
dataset = TranslationDataset(file_path)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
model = TranslationQualityModel()
trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=10, callbacks=model.configure_callbacks())
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
