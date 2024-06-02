from sentence_transformers import SentenceTransformer
import torch
import tensorflow_hub as hub
import numpy as np
from transformers import AutoTokenizer, AutoModel


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


