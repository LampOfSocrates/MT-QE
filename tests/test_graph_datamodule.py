import unittest
import torch
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from src.sent2graph import Sentence2Graph  # Adjust this import based on your file structure
from src.embedder_glove import GloveEmbedder

from src.embedder_glove import GLOVE_SIZE

class TestTextGraphDataset(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sentences = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog lies down"
        ]
        cls.dataset = Sentence2Graph(cls.sentences, embed_dim=GLOVE_SIZE)

    def test_initialization(self):
        self.assertEqual(len(self.dataset.sentences), 2)
        self.assertIsInstance(self.dataset.embedder, GloveEmbedder)

    def test_word_to_node_features(self):
        known_word = "quick"
        unknown_word = "asdflkjasdflkjasdf"

        known_embedding = self.dataset.word_to_node_features(known_word)
        unknown_embedding = self.dataset.word_to_node_features(unknown_word)

        self.assertEqual(known_embedding.shape, (GLOVE_SIZE,))
        self.assertTrue(torch.is_tensor(known_embedding))
        self.assertNotEqual(known_embedding.sum().item(), 0)

        self.assertEqual(unknown_embedding.shape, (GLOVE_SIZE,))
        self.assertTrue(torch.is_tensor(unknown_embedding))
        self.assertEqual(unknown_embedding.sum().item(), 0)

    def test_sentence_to_graph(self):
        sentence = "The quick brown fox"
        graph, _ = self.dataset.sentence_to_graph(sentence)

        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape, (4, GLOVE_SIZE))
        self.assertEqual(graph.edge_index.shape, (2, 3))

    def test_get_data(self):
        data_list = self.dataset.get_data()
        self.assertEqual(len(data_list), 2)
        for data in data_list:
            self.assertIsInstance(data, Data)

if __name__ == "__main__":
    unittest.main()
