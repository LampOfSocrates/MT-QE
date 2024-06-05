import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from embedders import TransformerEmbedder
from embedder_wordnet import WordNetGloveEmbedder
import pandas as pd
from torch.utils.data import Dataset
from eda import display_stats

class EmbeddedDataset(Dataset):
    def __init__(self, file_path, embedder , max_words_in_sentence=50, lp=None) :
        self.data = pd.read_csv(file_path)
        if lp:
            self.data = self.data[self.data == lp]
        self.embedder = embedder
        self.clean_data(max_words_in_sentence)

    def clean_data(self, max_words_in_sentence=50):
        ''' So in the WMT2022, we have at least 1 sample where the MT is nan . So we filter for nans
            Secondly during eda we found that 90% of the data has less than 50 words , 99% of the data has than 100 words
            So we normally work with smaller number of words in each sentence
            This translates into into less tokens for sentence and eventually much faster speed of training 
            than the whole dataset sceanrio where we pad everything up to 233 tokens each 
        '''
        data = self.data
        print(data.shape)
        data = data.dropna(subset=['src','mt', 'ref'])
        print(data.shape)
        def compute_length(text):
            if pd.isna(text):
                return 0
            return len(text.split()) 

        # Compute lengths of `src`, `mt`, and `ref` fields
        data['src_length'] = data['src'].apply(compute_length)
        data['mt_length'] = data['mt'].apply(compute_length)
        data['ref_length'] = data['ref'].apply(compute_length)
        data['max_length'] = data[['src_length', 'mt_length', 'ref_length']].max(axis=1)  # Longest sentence we have across src , mt and ref 
        data = data[data.max_length < max_words_in_sentence]
        print(data.shape)
        self.data = data
    
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
        # Load the full dataset
        full_dataset = EmbeddedDataset(self.file_path, self.embedder)
        
        # Calculate lengths for each split
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)
    


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