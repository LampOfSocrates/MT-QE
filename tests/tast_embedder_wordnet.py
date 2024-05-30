import unittest
import numpy as np
from nltk.tokenize import word_tokenize
from src.embedder_wordnet import WordNetSentenceEmbedder  # Adjust this import as necessary based on your file structure

class TestWordNetSentenceEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = WordNetSentenceEmbedder(language='eng', embedding_model_name='glove-wiki-gigaword-300')

    def test_embed_non_empty_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog"
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertNotEqual(embedding.sum(), 0)

    def test_embed_empty_sentence(self):
        sentence = ""
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertEqual(embedding.sum(), 0)

    def test_embed_sentence_with_unknown_words(self):
        sentence = "asdflkjasdflkjasdf"
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertEqual(embedding.sum(), 0)

    def test_embed_sentence_with_known_and_unknown_words(self):
        sentence = "The quick asdflkjasdflkjasdf"
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (300,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertNotEqual(embedding.sum(), 0)

if __name__ == "__main__":
    unittest.main()
