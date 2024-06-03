import torch 
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class TranslationDataset(Dataset):
    def __init__(self, file_path, max_words_in_sentence=50):
        self.data = pd.read_csv(file_path)
        self.clean_data(max_words_in_sentence=max_words_in_sentence)

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
        data.loc[:, 'src_length'] = data['src'].apply(compute_length)
        data.loc[:, 'mt_length'] = data['mt'].apply(compute_length)
        data.loc[:, 'ref_length'] = data['ref'].apply(compute_length)
        data.loc[:, 'max_length'] = data[['src_length', 'mt_length', 'ref_length']].max(axis=1)  # Longest sentence we have across src , mt and ref 
        data = data[data.max_length < max_words_in_sentence]
        print('Selected total data length', data.shape)
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'src': row['src'],
            'mt': row['mt'],
            'ref': row['ref'],
            'score': torch.tensor(row['score'], dtype=torch.float)
        }
