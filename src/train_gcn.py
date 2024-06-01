import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from nltk.tokenize import word_tokenize
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import gensim.downloader as api
from embedder_glove import GloveEmbedder
from sent2graph import GraphDataModule, Sentence2Graph
from gcn import GCN
from pytorch_lightning.loggers import WandbLogger
import wandb
from embedder_glove import GLOVE_SIZE

# Example sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog lies down"
]

# Prepare the data module
data_module = GraphDataModule(sentences, batch_size=2)

# Initialize the model
model = GCN(in_channels=300, hidden_channels=128, out_channels=64)

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',  # Monitor training loss
    dirpath='checkpoints/',  # Directory to save checkpoints
    filename='gcn-{epoch:02d}-{train_loss:.2f}',  # Filename pattern
    save_top_k=1,  # Save only the best model
    mode='min'  # Minimize the monitored quantity
)
# Initialize the WandbLogger
wandb_logger = WandbLogger(project='gcn-text-embeddings')

# Initialize the trainer with the checkpoint callback and WandbLogger
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback], logger=wandb_logger)

# Train the model
trainer.fit(model, data_module)

#########################################################
# Load the trained model from the checkpoint
checkpoint_path = checkpoint_callback.best_model_path  # Use the best model path
model = GCN.load_from_checkpoint(checkpoint_path, in_channels=300, hidden_channels=128, out_channels=64)

# New sentences to generate embeddings for
new_sentences = [
    "A quick brown fox",
    "A lazy dog"
]

# Initialize the TextGraphDataset for the new sentences
dataset = Sentence2Graph(new_sentences, embed_dim=GLOVE_SIZE)
graph_data_list = dataset.get_data()

# Generate embeddings for new sentences
model.eval()  # Set the model to evaluation mode

def generate_embedding(model, graph_data):
    with torch.no_grad():
        node_embeddings = model(graph_data)
    sentence_embedding = node_embeddings.mean(dim=0)
    return sentence_embedding

# Generate and print embeddings for each new sentence
for graph_data in graph_data_list:
    embedding = generate_embedding(model, graph_data)
    print(f"Sentence embedding: {embedding}")


