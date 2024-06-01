import gensim.downloader as api
import torch
import numpy as np

class GloveEmbedder:
    def __init__(self, model_name='glove-wiki-gigaword-300'):
        self.model = api.load(model_name)
    
    def embed(self, text):
        words = text.split()
        embeddings = [self.model[word] for word in words if word in self.model]
        if embeddings:
            return torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float)
        else:
            return torch.zeros(self.model.vector_size)
            
