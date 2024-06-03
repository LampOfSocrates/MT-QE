import os
import wandb
import torch 

from common import ROOT_FOLDER
# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"




from embedders import TransformerEmbedder
from embedded_dataset import EmbeddedLitModule
from pl_model_transformer import TranslationQualityModel
import pytorch_lightning as pl
from lightning.pytorch import  seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.loggers import WandbLogger

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

seed_everything(42, workers=True)

print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print the number of GPUs available
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU (if available)

# Initialize the embedder
embedder = TransformerEmbedder()

# Initialize the data module
data_module = EmbeddedLitModule(file_path='data/2022-da.csv', embedder=embedder, batch_size=64)

# Define the model
input_dim = 3 * 768  # assuming the embeddings are of size 768
hidden_dim = 256
output_dim = 1

model = TranslationQualityModel(input_dim, hidden_dim, output_dim)

print(model)
summary = pl.utilities.model_summary.ModelSummary(model, max_depth=2)
print(summary)
wandb.login(key=os.environ['WANDB_API_KEY'])
# Set the wandb directory to /tmp/wandb
os.environ['WANDB_DIR'] = f'{ROOT_FOLDER}/model1/wandb'

# Initialize wandb
wandb.init(project='model2', dir=os.environ['WANDB_DIR'])


wandb_logger = WandbLogger(project='transformer-text-embeddings-heron')
profiler = AdvancedProfiler(dirpath=f"{ROOT_FOLDER}/model1/profiler", filename="perf_logs")

# Initialize the trainer
# auto uses gpu if available
trainer = pl.Trainer(default_root_dir=f"{ROOT_FOLDER}/model1/",
                    max_epochs=10,
                    devices=-1,
                    accelerator="gpu", 
                    callbacks=model.configure_callbacks(),
                    logger=wandb_logger,
                    #check_val_every_n_epoch=1,
                    #profiler=profiler,
                    precision="16-mixed",
                    fast_dev_run=10,                            # Just run 7 batches 
                    log_every_n_steps=10,                       # Because we are running only 10 batches
                    limit_train_batches=0.01, limit_val_batches=0.01 , limit_test_batches=0.01 #use 10 batches of train and 5 batches of val and 10% of test data
                    ) 

wandb_logger.watch(model)

# Train the model
trainer.fit(model, datamodule=data_module)

# Optionally test the model
trainer.test(model, datamodule=data_module)
