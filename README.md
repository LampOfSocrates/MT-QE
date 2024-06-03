
# Intro 

* This experiments with Machine Transaltion datasets for a Quality Evaluation Task . 
* It takes in source and machine translated outputs ( along with a reference if available ).
* It then creates embeddings for these using a variety of methods 
* It also constructs a Graph CNN from the text , trains the GCN to arrive a new set of embeddings
* Largely using Torch Lighning 
* Used cloned code from other repos where necessary with copyright notices intact 
* Pushes models to HF
* Pushes metrics to Wandb
* Uses Github Actions for Test Coverage 

# References 

* Zhao, Haofei, et al. "From Handcrafted Features to LLMs: A Brief Survey for Machine Translation Quality Estimation." arXiv preprint arXiv:2403.14118 (2024).

* Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A Neural Framework for MT Evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685–2702, Online. Association for Computational Linguistics.

# Annotation Methods
* HTER - computes the translation edit rate TER between mt and pe . 

# Datasets
 * QT21 - 4 language pairs, 173K tuples of ( src, mt, ref , pe )   - pe means post edited https://lindat.mff.cuni.cz/ repository/xmlui/handle/11372/LRT-2390
 * WMT Shared Tasks 2023 - 14 language pairs with PE, 17 with DA , 4 with MQM  . raw: http://www.statmt.org/wmt19/results.html 
 * MLQE-PE 2020 - 11 language Pairs
 * FLORES 2019 

# Download NLP data
## Download Glove Embeddings
Use glove-wiki-gigaword-300 everywhere for better results than glove-wiki-gigaword-50

## Download wordnet and nltlk and spacy gunk
Additional downloads 
<code>
import nltk
nltk.download('wordnet')
nltk.download('punkt')
</code>

## Download Spacy
python -m spacy download en_core_web_sm

# Testing 

## Exceute locally 
python -m unittest discover -s tests

or for individual tests

python -m unittest tests.test_sent2graph
python -m unittest tests.test_embedder_wordnet
python -m unittest tests.test_gcn_train

Check for mocking in each test separately 

## Run tests with coverage
coverage run -m unittest discover -s tests
coverage report
coverage html

## Coverall report 

Here Github Actions pumps the test coverage to a site

[![Coverage Status](https://coveralls.io/repos/github/LampOfSocrates/MT-QE/badge.svg?branch=main)](https://coveralls.io/github/LampOfSocrates/MT-QE?branch=main)

## TODO
1) Pad and align tokens 
2) Check tokenization quality per language
3) Instead of concat 3 input sets of token , apply COMET like operations
4) record the difference between src+mt+ref vs mt-ref in terms of regression score
5) Train only specific language pairs
