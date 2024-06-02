import torch
import tensorflow_hub as hub
import gensim.downloader as api
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
import torch


class FastTextEmbedder:
    def __init__(self, model_path='cc.en.300.bin'):
        self.model = fasttext.load_model(model_path)
    
    def embed(self, text):
        return torch.tensor(self.model.get_sentence_vector(text), dtype=torch.float)

class Word2VecEmbedder:
    def __init__(self, model_name='word2vec-google-news-300'):
        self.model = api.load(model_name)
    
    def embed(self, text):
        words = text.split()
        embeddings = [self.model[word] for word in words if word in self.model]
        if embeddings:
            return torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float)
        else:
            return torch.zeros(self.model.vector_size)


class TfidfEmbedder:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)
    
    def embed(self, text):
        return torch.tensor(self.vectorizer.transform([text]).toarray(), dtype=torch.float).squeeze()