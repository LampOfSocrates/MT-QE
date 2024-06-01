
# Intro 

This experiments with Machine Transaltion datasets for a Quality Evaluation Task . 
It takes in source and machine translated outputs ( along with a reference if available ).
It then creates embeddings for these using a variety of methods 
It also constructs a Graph CNN from the text , trains the GCN to arrive a new set of embeddings

# Glove Embeddings
Use glove-wiki-gigaword-300 everywhere for better results than glove-wiki-gigaword-50

# Download wordnet and nltlk and spacy gunk
import nltk
nltk.download('wordnet')

# Addituonal downloads
python -m spacy download en_core_web_sm

## Run tests
python -m unittest discover -s tests

or for individual tests

python -m unittest tests.test_sent2graph

Check for mocking in each test separately 

# Run tests with coverage
coverage run -m unittest discover -s tests
coverage report
coverage html