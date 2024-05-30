import unittest
import torch
import numpy as np
from src.embedders import GloveEmbedder
from graph_datamodule import GraphDataModule

class TestTrainGCN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = GloveEmbedder()

    def test_gcn_training_non_empty_sentence(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog lies down"
        ]

        # Prepare the data module
        data_module = GraphDataModule(sentences, batch_size=2)

        # Initialize the model
        model = GCN(in_channels=300, hidden_channels=128, out_channels=64)

        # Initialize the trainer
        trainer = pl.Trainer(max_epochs=1)

        # Train the model
        trainer.fit(model, data_module)

if __name__ == "__main__":
    unittest.main()