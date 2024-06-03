import os 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.utils.data import random_split
from model3 import TranslationQualityModel
import pytorch_lightning as pl
import torch 
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from lightning.pytorch import  seed_everything
from pytorch_lightning.strategies import DDPStrategy
from common import ROOT_FOLDER
from dat_wmt import TranslationDataset
# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


seed_everything(42, workers=True)

print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print the number of GPUs available
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU (if available)

######### LOAD DATA #############################
file_path = 'data/2022-da.csv'
dataset = TranslationDataset(file_path, max_words_in_sentence=50)

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

######### LOAD MODEL  #############################

backbone_name = 'xlm-roberta-base'
model = TranslationQualityModel(model_name=backbone_name)

######### Setup WandB  #############################
wandb.login(key=os.environ['WANDB_API_KEY'])
# Set the wandb directory to /tmp/wandb
os.environ['WANDB_DIR'] = f'{ROOT_FOLDER}/model3/wandb'

config = {"lr": 3e-4, "batch_size": 8}
config.update({"architecture": backbone_name, "max_words_in_sentence": 50})

# Initialize wandb
wandb.finish()
wandb.init(project='model3', dir=os.environ['WANDB_DIR'], config=config)

wandb_logger = WandbLogger(project='model3-text-embeddings-heron')

######### Setup Training  #############################
profiler = SimpleProfiler(dirpath=f"{ROOT_FOLDER}/model3/profiler", filename="perf_logs")

trainer = pl.Trainer(default_root_dir=f"{ROOT_FOLDER}/model3/", 
                     accelerator='gpu', 
                     devices=-1, 
                     #strategy=DDPStrategy(find_unused_parameters=False),
                     #num_nodes=2,  # Number of machines (nodes)
                     max_epochs=10, 
                     callbacks=model.configure_callbacks(),
                     profiler=profiler,
                     logger=wandb_logger,
                     limit_train_batches=1.0, limit_val_batches=1.0 , limit_test_batches=1.0 #use 10 batches of train and 5 batches of val and 10% of test data
                     )

######### Train and Test #############################
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
wandb.finish()