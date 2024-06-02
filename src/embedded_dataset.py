import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from embedders import TransformerEmbedder
from embedder_wordnet import WordNetGloveEmbedder
import pandas as pd
from torch.utils.data import Dataset

class EmbeddedDataset(Dataset):
    def __init__(self, file_path, embedder):
        self.data = pd.read_csv(file_path)
        self.embedder = embedder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        lp = row['lp']
        src = row['src']
        mt = row['mt']
        ref = row['ref']
        score = row['score']
        raw = row['raw']
        annotators = row['annotators']
        domain = row['domain']
        
        # Generate embeddings using the encoder
        src_embedding = self.embedder.embed(src)
        mt_embedding = self.embedder.embed(mt)
        ref_embedding = self.embedder.embed(ref)
        
        sample = {
            'lp': lp,
            'src_embedding': src_embedding,
            'mt_embedding': mt_embedding,
            'ref_embedding': ref_embedding,
            'score': torch.tensor(score, dtype=torch.float),
            'raw': torch.tensor(raw, dtype=torch.float),
            'annotators': torch.tensor(annotators, dtype=torch.int),
            'domain': domain
        }
        
        return sample

class EmbeddedLitModule(pl.LightningDataModule):
    def __init__(self, file_path, embedder, batch_size=32):
        super().__init__()
        self.file_path = file_path
        self.embedder = embedder
        self.batch_size = batch_size
        
    
    def setup(self, stage=None):
        self.dataset = EmbeddedDataset(self.file_path, self.embedder)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=7)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=7)

def display_stats(data):

    # Compute the unique count of the `lp` field
    unique_lp_count = data['lp'].nunique()

    print(data.head(2))

    # Function to compute the length of text
    def compute_length(text):
        if pd.isna(text):
            return 0
        return len(text.split()) 

    # Compute lengths of `src`, `mt`, and `ref` fields
    data['src_length'] = data['src'].apply(compute_length)
    data['mt_length'] = data['mt'].apply(compute_length)
    data['ref_length'] = data['ref'].apply(compute_length)

    # Group by `lp` field and calculate required statistics
    grouped = data.groupby('lp').agg({
        'src_length': ['count', 'min', 'max', 'mean'],
        'mt_length': ['min', 'max', 'mean'],
        'ref_length': ['min', 'max', 'mean'],
        'annotators': 'mean',
        'raw': 'mean'
    }).reset_index()

    # Flatten the column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Rename columns for clarity
    grouped.rename(columns={
        'lp_': 'lp',
        'src_length_min': 'src_min_length',
        'src_length_max': 'src_max_length',
        'src_length_mean': 'src_avg_length',
        'mt_length_min': 'mt_min_length',
        'mt_length_max': 'mt_max_length',
        'mt_length_mean': 'mt_avg_length',
        'ref_length_min': 'ref_min_length',
        'ref_length_max': 'ref_max_length',
        'ref_length_mean': 'ref_avg_length',
        'annotators_mean': 'avg_annotators',
        'raw_mean': 'avg_raw_score'
    }, inplace=True)

    # Print the results
    print(f'Unique count of `lp` field: {unique_lp_count}')
    print(grouped)

    # Display the dataframe in a readable format
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', True):
        print(grouped)

def main():
# Example usage:
    file_path = 'data/2022-da.csv'
    data = pd.read_csv(file_path)
    display_stats(data)
    # Create encoders
    transformer_embedder = TransformerEmbedder()
    wordnet_embeder = WordNetGloveEmbedder()
    #word2vec_encoder = Word2VecEncoder()
    #fasttext_encoder = FastTextEncoder()
    #use_encoder = USEEncoder()

    # Inject the encoder into the EmbeddedDataset
    dataset_with_transformer = EmbeddedDataset(file_path, transformer_embedder)
    dataset_with_wordnet_embeddings = EmbeddedDataset(file_path, wordnet_embeder)
    #dataset_with_tfidf = EmbeddedDataset(file_path, tfidf_encoder)
    #dataset_with_word2vec = EmbeddedDataset(file_path, word2vec_encoder)
    #dataset_with_fasttext = EmbeddedDataset(file_path, fasttext_encoder)
    #dataset_with_use = EmbeddedDataset(file_path, use_encoder)

    BATCH_SIZE = 2
    dataloader_with_transformer = DataLoader(dataset_with_transformer, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_with_wordnet_embeddings = DataLoader(dataset_with_wordnet_embeddings, batch_size=BATCH_SIZE, shuffle=True)
    #dataloader_with_use = DataLoader(dataset_with_use, batch_size=BATCH_SIZE, shuffle=True)
    #dataloader_with_word2vec = DataLoader(dataset_with_word2vec, batch_size=32, shuffle=True)
    #dataloader_with_fasttext = DataLoader(dataset_with_fasttext, batch_size=BATCH_SIZE, shuffle=True)

    for dat in dataloader_with_transformer:
        print(dat)
        break
    


if __name__ == '__main__':
    main()