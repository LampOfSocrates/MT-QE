from unittest.mock import MagicMock, patch
import unittest
import numpy as np
from src.embedder_wordnet import WordNetGloveEmbedder, GLOVE_SIZE  # Adjust this import as necessary based on your file structure

class TestWordNetGloveEmbedder(unittest.TestCase):

    @patch('gensim.models.KeyedVectors.load_word2vec_format')
    def setUp(self, mock_glove_load):
        self.mock_model = MagicMock()
        # Mock the behavior of the model
        self.mock_model.__contains__.side_effect = lambda word: word != "asdflkjasdflkjasdf"
        self.mock_model.__getitem__.side_effect = lambda word: np.random.rand(GLOVE_SIZE) if word != "asdflkjasdflkjasdf" else np.zeros(GLOVE_SIZE)
        mock_glove_load.return_value = self.mock_model

        # Initialize the embedder with the mocked GloVe model
        self.embedder = WordNetGloveEmbedder(language='eng', embedding_model_name='glove-wiki-gigaword-50')

    def test_embed_non_empty_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog"
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (GLOVE_SIZE,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertNotEqual(embedding.sum(), 0)

    def test_embed_empty_sentence(self):
        sentence = ""
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (GLOVE_SIZE,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertEqual(embedding.sum(), 0)

    def test_embed_sentence_with_unknown_words(self):
        sentence = "asdflkjasdflkjasdf"
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (GLOVE_SIZE,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertEqual(embedding.sum(), 0)

    def test_embed_sentence_with_known_and_unknown_words(self):
        sentence = "The quick asdflkjasdflkjasdf"
        embedding = self.embedder.embed_sentence(sentence)
        self.assertEqual(embedding.shape, (GLOVE_SIZE,))
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertNotEqual(embedding.sum(), 0)

if __name__ == "__main__":
    unittest.main()
