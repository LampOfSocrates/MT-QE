import unittest
import torch
import numpy as np
from src.embedder_glove import GloveEmbedder , GLOVE_SIZE
from src.sent2graph import GraphDataModule
from src.gcn import GCN
import pytorch_lightning as pl

class TestTrainGCN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = GloveEmbedder(model_name= 'glove-wiki-gigaword-50'   )

    def test_gcn_training_non_empty_sentence(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog lies down"
        ]

        # Prepare the data module
        data_module = GraphDataModule(sentences, batch_size=2)

        # Initialize the model
        model = GCN(in_channels=GLOVE_SIZE, hidden_channels=128, out_channels=64)

        # Initialize the trainer
        trainer = pl.Trainer(max_epochs=1)

        # Train the model
        trainer.fit(model, data_module)

    def test_transformer_embedded(self):

        #datamodule = EmbeddedLitModule(file_path, encoder, batch_size=32)
        pass

if __name__ == "__main__":
    unittest.main()