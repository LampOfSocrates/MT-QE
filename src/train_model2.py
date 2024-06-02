import os 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.utils.data import random_split
from model2 import TranslationQualityModel
import pytorch_lightning as pl
import torch 
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from lightning.pytorch import  seed_everything
from pytorch_lightning.strategies import DDPStrategy

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


seed_everything(42, workers=True)

print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print the number of GPUs available
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU (if available)


file_path = 'data/2022-da.csv'
dataset = TranslationDataset(file_path)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

print("train size: ", train_size)
print("val size: ", val_size)
print("test size: ", test_size)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, num_workers=2)

model = TranslationQualityModel()

wandb.login(key=os.environ['WANDB_API_KEY'])

wandb_logger = WandbLogger(project='transformer-text-embeddings-heron')
profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

trainer = pl.Trainer(accelerator='gpu', 
                     devices=-1, 
                     #strategy=DDPStrategy(find_unused_parameters=False),
                     #num_nodes=2,  # Number of machines (nodes)
                     max_epochs=10, 
                     callbacks=model.configure_callbacks(),
                     profiler=profiler,
                     logger=wandb_logger,
                     limit_train_batches=1.0, limit_val_batches=1.0 , limit_test_batches=1.0 #use 10 batches of train and 5 batches of val and 10% of test data
                     )
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
