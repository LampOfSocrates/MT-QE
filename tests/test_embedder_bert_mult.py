import unittest
import torch
from src.embedders import TransformerEmbedder  # Adjust this import as necessary based on your file structure

class TestTransformerEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = TransformerEmbedder(model_name='bert-base-multilingual-cased')

    def test_embed_non_empty_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog"
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertNotEqual(embedding.sum().item(), 0)

    def test_embed_empty_sentence(self):
        sentence = ""
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertNotEqual(embedding.sum().item(), 0)

    def test_embed_sentence_with_unknown_words(self):
        sentence = "asdflkjasdflkjasdf"
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertNotEqual(embedding.sum().item(), 0)

    def test_embed_sentence_with_known_and_unknown_words(self):
        sentence = "The quick asdflkjasdflkjasdf"
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (768,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertNotEqual(embedding.sum().item(), 0)

if __name__ == "__main__":
    unittest.main()
