
# Intro 

This experiments with Machine Transaltion datasets for a Quality Evaluation Task . 
It takes in source and machine translated outputs ( along with a reference if available ).
It then creates embeddings for these using a variety of methods 
It also constructs a Graph CNN from the text , trains the GCN to arrive a new set of embeddings

## Run tests
python -m unittest discover -s tests

or

python tests/test_text_graph_dataset.py