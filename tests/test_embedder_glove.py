import unittest
import torch
import numpy as np
from embedder_glove import GloveEmbedder

class TestGloveEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = GloveEmbedder()

    def test_embed_non_empty_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog"
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertNotEqual(embedding.sum().item(), 0)

    def test_embed_empty_sentence(self):
        sentence = ""
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertEqual(embedding.sum().item(), 0)

    def test_embed_sentence_with_unknown_words(self):
        sentence = "asdflkjasdflkjasdf"
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertEqual(embedding.sum().item(), 0)

    def test_embed_sentence_with_known_and_unknown_words(self):
        sentence = "The quick asdflkjasdflkjasdf"
        embedding = self.embedder.embed(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(torch.is_tensor(embedding))
        self.assertNotEqual(embedding.sum().item(), 0)

if __name__ == "__main__":
    unittest.main()