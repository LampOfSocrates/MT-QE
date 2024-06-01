import unittest
import torch
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from src.sent2graph import Sentence2Graph  

class TestSentence2Graph(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sentences = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog lies down"
        ]
        cls.sent2graph = Sentence2Graph(cls.sentences)

    def test_print_graph(self):
        sentence = "The quick brown fox jumps over the lazy dog."
        self.sent2graph.print_adjacency_matrix(sentence)
        
    
if __name__ == "__main__":
    unittest.main()
