import os

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from embedders import TransformerEmbedder
from embedded_dataset import EmbeddedLitModule
from pl_model_transformer import TranslationQualityModel
import pytorch_lightning as pl
import torch 
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

seed_everything(42, workers=True)

# Initialize the embedder
embedder = TransformerEmbedder()

# Initialize the data module
data_module = EmbeddedLitModule(file_path='data/2022-da.csv', embedder=embedder, batch_size=32)

# Define the model
input_dim = 3 * 768  # assuming the embeddings are of size 768
hidden_dim = 256
output_dim = 1

model = TranslationQualityModel(input_dim, hidden_dim, output_dim)

# Initialize the trainer
# auto uses gpu if available
trainer = pl.Trainer(max_epochs=1, 
                    accelerator="auto", 
                    callbacks=model.configure_callbacks(),
                    check_val_every_n_epoch=1)

# Train the model
trainer.fit(model, datamodule=data_module)

# Optionally test the model
trainer.test(model, datamodule=data_module)
