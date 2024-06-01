import gensim.downloader as api
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GLOVE_SIZE=50

class GloveEmbedder:
    def __init__(self, model_name='glove-wiki-gigaword-50'):
        logging.info("Starting Glove loading")
        self.model = api.load(model_name)
    
    def embed(self, text):
        words = text.split()
        embeddings = [self.model[word] for word in words if word in self.model]
        if embeddings:
            return torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float)
        else:
            return torch.zeros(self.model.vector_size)
            
