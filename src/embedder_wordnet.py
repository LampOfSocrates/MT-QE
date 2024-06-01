from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from src.embedder_glove import GLOVE_SIZE , GloveEmbedder
import numpy as np
import nltk
import gensim.downloader as api

class MultilingualWordNetEncoder:
    def __init__(self, language='eng'):
        self.language = language

    def get_synsets(self, word):
        """
        Get synsets for a given word in the specified language.
        """
        synsets = wn.synsets(word, lang=self.language)
        return synsets

    def get_synset_features(self, word):
        """
        Get features from synsets for a given word.
        """
        synsets = self.get_synsets(word)
        features = []
        for synset in synsets:
            # Example features: lemma names, definition, hypernyms, hyponyms
            lemmas = synset.lemma_names(self.language)
            definition = synset.definition()
            hypernyms = synset.hypernyms()
            hyponyms = synset.hyponyms()
            # You can add more features as needed

            features.append({
                'lemmas': lemmas,
                'definition': definition,
                'hypernyms': [hypernym.name() for hypernym in hypernyms],
                'hyponyms': [hyponym.name() for hyponym in hyponyms]
            })
        return features

    def encode(self, word):
        """
        Encode the given word using WordNet features.
        """
        synset_features = self.get_synset_features(word)
        return synset_features


# TODO use the GloveEmbedder class
class WordNetGloveEmbedder:
    def __init__(self, language='eng', embedding_model_name='glove-wiki-gigaword-50'):
        self.encoder = MultilingualWordNetEncoder(language)
        self.model = GloveEmbedder(model_name=embedding_model_name).model

    def get_word_embedding(self, word):
        """
        Get the embedding for a given word using a pre-trained embedding model.
        """
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(GLOVE_SIZE)  # Return a zero vector if the word is not in the embedding model

    def embed_synset_features(self, features):
        """
        Convert synset features into a single embedding vector.
        """
        lemma_embeddings = [self.get_word_embedding(lemma) for lemma in features['lemmas']]
        hypernym_embeddings = [self.get_word_embedding(hypernym.split('.')[0]) for hypernym in features['hypernyms']]
        hyponym_embeddings = [self.get_word_embedding(hyponym.split('.')[0]) for hyponym in features['hyponyms']]

        # Aggregate all embeddings (e.g., by averaging)
        all_embeddings = lemma_embeddings + hypernym_embeddings + hyponym_embeddings
        if all_embeddings:
            return np.mean(all_embeddings, axis=0)
        else:
            return np.zeros(GLOVE_SIZE)

    def embed_word(self, word):
        synset_features = self.encoder.encode(word)
        if not synset_features:
            return np.zeros(GLOVE_SIZE)  # Return a zero vector if no synsets are found

        # Aggregate synset feature embeddings
        synset_embeddings = [self.embed_synset_features(features) for features in synset_features]
        word_embedding = np.mean(synset_embeddings, axis=0) if synset_embeddings else np.zeros(GLOVE_SIZE)
        return word_embedding

    def embed_sentence(self, sentence):
        words = word_tokenize(sentence)
        word_embeddings = [self.embed_word(word) for word in words]

        # Aggregate word embeddings to form a sentence embedding (e.g., by averaging)
        sentence_embedding = np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(GLOVE_SIZE)
        return sentence_embedding

