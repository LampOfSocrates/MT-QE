from src.common import measure_import_time

with measure_import_time('your_module_name'):
    from unittest.mock import MagicMock, patch
    import unittest
    import torch
    from src.sent2graph import Sentence2Graph  
    import logging
    from src.embedder_glove import GloveEmbedder, GLOVE_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class TestSentence2Graph(unittest.TestCase):
    
    def mock_spacy(self):
        mock_nlp = MagicMock()
        mock_doc = MagicMock()

        # Create mock tokens
        mock_token1 = MagicMock()
        mock_token1.text = "The"
        mock_token1.i = 0
        mock_token1.children = []

        mock_token2 = MagicMock()
        mock_token2.text = "quick"
        mock_token2.i = 1
        mock_token2.children = [mock_token1]

        mock_token3 = MagicMock()
        mock_token3.text = "brown"
        mock_token3.i = 2
        mock_token3.children = [mock_token2]

        mock_token4 = MagicMock()
        mock_token4.text = "fox"
        mock_token4.i = 3
        mock_token4.children = [mock_token3]

        # Set the mock doc to return the mock tokens
        mock_doc.__iter__.return_value = [mock_token1, mock_token2, mock_token3, mock_token4]

        # Configure the mock nlp to return the mock doc
        mock_nlp.return_value = mock_doc

        return mock_nlp

    
    @classmethod
    @patch('gensim.models.KeyedVectors.load_word2vec_format')
    def setUpClass(cls, mock_glove_load):
        logging.info("Starting Setup")
        mock_glove_load.return_value = MagicMock()
        cls.sentences = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog lies down"
        ]
        cls.sent2graph = Sentence2Graph(cls.sentences, embed_dim=GLOVE_SIZE)

    @patch('spacy.load')
    def test_print_graph(self,  mock_spacy_load):
        logging.info("Starting test")
        sentence = "The quick brown fox jumps over the lazy dog."

        mock_nlp = self.mock_spacy()
        mock_spacy_load.return_value = mock_nlp
        self.sent2graph.print_adjacency_matrix(sentence)
        
    
if __name__ == "__main__":
    unittest.main()
