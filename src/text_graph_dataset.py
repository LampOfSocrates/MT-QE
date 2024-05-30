import torch
from torch_geometric.data import Data, DataLoader
from nltk.tokenize import word_tokenize
import pytorch_lightning as pl
from embedders import GloveEmbedder

class TextGraphDataset:
    def __init__(self, sentences, embedder=GloveEmbedder()):
        self.sentences = sentences
        self.embedder = embedder # Initialize the GloveEmbedder

    def word_to_node_features(self, word):
        return self.embedder.embed(word).float()

    def sentence_to_graph(self, sentence):
        words = word_tokenize(sentence.lower())
        edge_index = []
        node_features = []

        for i, word in enumerate(words):
            node_features.append(self.word_to_node_features(word))
            if i > 0:
                edge_index.append([i-1, i])
                edge_index.append([i, i-1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.stack(node_features)
        data = Data(x=node_features, edge_index=edge_index)
        return data

    def get_data(self):
        return [self.sentence_to_graph(sentence) for sentence in self.sentences]

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, sentences, batch_size=32):
        super().__init__()
        self.sentences = sentences
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = TextGraphDataset(self.sentences)
        self.data = dataset.get_data()

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)
