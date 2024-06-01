import torch
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from nltk.tokenize import word_tokenize
import pytorch_lightning as pl
from src.embedder_glove import GloveEmbedder

class Sentence2Graph:
    def __init__(self, sentences, embedder=GloveEmbedder()):
        self.sentences = sentences
        self.embedder = embedder # Initialize the GloveEmbedder
        print("Starting Space Load")
        self.nlp = spacy.load('en_core_web_sm')
        self.deptree = True
        print("Loaded Sentence2Graph ")
        
    def word_to_node_features(self, word):
        return self.embedder.embed(word).float()

    def sentence_to_graph(self, sentence):
        doc = self.nlp(sentence.lower())
        words = word_tokenize(sentence.lower())
        edge_index = []
        node_features = []
        
        if self.deptree:
            for token in doc:
                node_features.append(self.word_to_node_features(token.text))
            for token in doc:
                for child in token.children:
                    edge_index.append([token.i, child.i])
        else:
            for i, word in enumerate(words):
                node_features.append(self.word_to_node_features(word))
                if i > 0:
                    edge_index.append([i-1, i])
                    edge_index.append([i, i-1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.stack(node_features)
        data = Data(x=node_features, edge_index=edge_index)
        return data

    def visualize_graph(self, sentence):
        data, doc = self.sentence_to_graph(sentence)

        # Create a NetworkX graph from the edge_index
        G = nx.DiGraph()
        for i in range(data.edge_index.shape[1]):
            source = data.edge_index[0, i].item()
            target = data.edge_index[1, i].item()
            G.add_edge(source, target)

        # Get node labels
        labels = {i: token.text for i, token in enumerate(doc)}

        # Draw the graph
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrows=True)
        plt.show()

    def edge_index_to_adjacency_matrix(self, edge_index, num_nodes):
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(edge_index.shape[1]):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            adj_matrix[source, target] = 1
        return adj_matrix
    
    def print_adjacency_matrix(self, sentence):
        data, doc = self.sentence_to_graph(sentence)
        adj_matrix = self.edge_index_to_adjacency_matrix(data.edge_index, len(doc))
        
        print("Adjacency Matrix:")
        print(adj_matrix)

    def get_data(self):
        return [self.sentence_to_graph(sentence) for sentence in self.sentences]

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, sentences, batch_size=32):
        super().__init__()
        self.sentences = sentences
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = Sentence2Graph(self.sentences)
        self.data = dataset.get_data()

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

