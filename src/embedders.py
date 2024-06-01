from sentence_transformers import SentenceTransformer
import torch
import tensorflow_hub as hub
import fasttext
import gensim.downloader as api
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoTokenizer, AutoModel
import torch

'''The TransformerEncoder and USEEncoder have the best multi lingual support '''

class TransformerEmbedder:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        ''' Can also use xlm-roberta-base
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()


class USEEmbedder:
    def __init__(self, model_url='https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'):
        self.model = hub.load(model_url)
    
    def embed(self, text):
        return torch.tensor(self.model([text]).numpy().squeeze(), dtype=torch.float)

class SentenceTransformerEmbedder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text):
        embeddings = self.model.encode(text, convert_to_tensor=True)
        return embeddings

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
