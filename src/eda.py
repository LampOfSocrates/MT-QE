import pandas as pd

def display_stats(data):

    # Compute the unique count of the `lp` field
    unique_lp_count = data['lp'].nunique()

    #print(data.head(2))

    # Function to compute the length of text
    def compute_length(text):
        if pd.isna(text):
            return 0
        return len(text.split()) 

    # Compute lengths of `src`, `mt`, and `ref` fields
    data['src_length'] = data['src'].apply(compute_length)
    data['mt_length'] = data['mt'].apply(compute_length)
    data['ref_length'] = data['ref'].apply(compute_length)
    data['max_length'] = data[['src_length', 'mt_length', 'ref_length']].max(axis=1)  # Longest sentence we have across src , mt and ref 

    # Group by `lp` field and calculate required statistics
    grouped = data.groupby('lp').agg({
        'src_length': ['count', 'min', 'max', 'mean'],
        'mt_length': ['min', 'max', 'mean'],
        'ref_length': ['min', 'max', 'mean'],
        'max_length': ['max'],
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
        'max_length': 'max_length_max',  
        'annotators_mean': 'avg_annotators',
        'raw_mean': 'avg_raw_score'
    }, inplace=True)

    # Print the results
    print(f'Unique count of `lp` field: {unique_lp_count}')
    #print(grouped)

    # Display the dataframe in a readable format
    #with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', True):
    #    print(grouped)

    return grouped, data


def lang_detector(text):
    # import packages
    import spacy
    from spacy.language import Language
    from spacy_langdetect import LanguageDetector
    # load english vocab and create pipeline 
    def get_lang_detector(nlp, name):
        return LanguageDetector()
        
    nlp = spacy.load("en_core_web_sm")
    Language.factory("language_detector", func=get_lang_detector)
    
    nlp.add_pipe('language_detector', last=True)
    # use created pipeline for language detect
    def detect_lan(text) :
        doc = nlp(text)
        detect_language = doc._.language 
        detect_language = detect_language['language']
        return(detect_language)

    detect_lan(text)