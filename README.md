# Label-GCN
A variation of GCN that allows the model to learn from labelled nodes in a graph.
Paper available at https://arxiv.org/abs/2104.02153.

## Overview

The implementation of Label-GCN relies on the functionality available through Keras and Stellargraph. The Stellargraph library has been
modified, with the main logic of Label-GCN contained in the file `label_gcn.py` located at `stellargraph/layer/label_gcn.py`. 
This modified Stellargraph library is available at https://github.com/cbellei/stellargraph and 
is used as a submodule in this project (see next section).

Unfortunately, Tensorflow does not easily support the use of sparse tensors within the `tf.Linalg` package (see here https://github.com/tensorflow/tensorflow/issues/27380 for some details); this
has the effect that currently the implementation provided in this project is inefficient for large graphs (such as the Elliptic dataset). 

## Installation
* Tested with Python 3.6
* Clone the repository and add the Stellargraph submodule, modified with the addition of Label-GCN
```
git clone https://github.com/cbellei/LabelGCN.git
cd LabelGCN
git submodule init
git submodule update
```
* Set up the environment (Anaconda): 
```
conda create -n LabelGCN python=3.6
conda activate LabelGCN
pip install -r requirements.txt
cd stellargraph
pip install -e .
``` 

## Datasets

The CORA, Citeseer and Pubmed datasets are available through the Stellargraph library. The Elliptic dataset is 
available at https://www.kaggle.com/ellipticco/elliptic-data-set. This project expects the Elliptic dataset to be located under a directory named `elliptic_bitcoin_dataset`.

## Running the experiments

The transductive experiments of Tables 3 and 4 can be produced running the file `experiments_transductive.py`. The dataset, number of random states
and number of runs for each random state are set by the flags `ds`, `ns` and `nr` respectively.
It is advisable to run with `ds=cora`, `ns=1` and `nr=1` for the quickest run. This would result in the following command:
```
python src/experiments_transductive.py -ds cora -ns 1 -nr 1
```
The inductive experiment of Table 5 can be produced running the file `experiments_inductive.py`. Three flags can be set in this case: `ns` determines the number of random states, as before, while `nr1` and `nr2` the number of runs for the standalone classical machine 
learning models and for the classical machine learning models with the addition of the GCN/LabelGCN embeddings. In the latter case,
for each set of embeddings (set via `ns`), a number of runs `nr2` is performed. For this experiment, the quickest run results from the command:
```
python src/experiments_inductive.py -ns 1 -nr1 1 -nr2 1
```

NOTE: All runs involving the Elliptic dataset are computationally demanding. Producing Tables 4 and 5 of the paper 
required running on a server for several hours.
