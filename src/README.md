# XOR: Python code

## Table of Contents  

- [Files](#files)  
- [Files in modules directory](#files-in-modules-directory)  
- [Files in scripts directory](#files-in-scripts-directory)  
- [Command-line usage](#command-line-usage)
    - [Parameters](#parameters)  
- [Full demo](#full-demo)  
- [I/O format](#io-format)  
    - [Input](#input)
    - [Output](#output)

## Files
- `setting_syn_XOR.yaml` : Setting to generate synthetic data (input for *generate_graph.py*).
- `setting_XOR.yaml` : Setting to run the algorithm XOR (input for *main.py*).
- `setting_MT.yaml` : Setting to run the algorithm MultiTensor (input for *main.py*).
- `setting_SR.yaml` : Setting to run the algorithm SpringRank (input for *main.py*).
- `XOR_demo.ipynb` : Example jupyter notebook of usage of the algorithm and import the output results.


## Files in `modules` directory
- `XOR_generator.py` : Class definition of the XOR generative model. It builds a directed,
  possibly weighted, network. It contains functions to generate networks with intrinsic community and hierarchical structures.
  The amount of influence on the network topology of the two structures is regulated by a scalar value in [0,1],
  corresponding to the expected portion of nodes preferring the hierarchical interaction mechanism.
- `XOR.py` : Class definition of the XOR inference model. It takes in input a single-layer network and infers the portion of nodes
  interaction via community structure, the portion interacting via hierarchical structure and the latent variable of the two mechanisms
  respectively.
- `MultiTensor.py` : Class definition of MultiTensor. Original implementation can be found [here](https://github.com/cdebacco/MultiTensor).
- `SpringRank.py` : Class definition of SpringRank. Original implementation can be found [here](https://github.com/cdebacco/SpringRank).
- `tools.py` : Contains non-class functions for handling the data.
- `compute_matrices.py` : Contains functions for computing and saving all the inference outputs in a cvs file.

## Files in `scripts` directory
- `main.py` : Code for inference with the selected algorithm. It performs the inference in the given single-layer directed network.
  It infers latent variables as community memberships and ranking for nodes, plus the node coefficients for the preferred interaction mechanism.
  If requested in the setting file, it performs a k-fold cross-validation procedure in order to estimate the hyperparameter **K**
  (number of communities). It runs with a given K and returns a csv file summarizing the results over each fold. The output file
  contains, among other information: the value of the log-likelihood, the ROC-AUC and the PR-AUC (for both classes) of type prediction
  both in the train and in test sets, the ROC-AUC for the link prediction both in the train and in test sets.
- `generate_graph.py` : Code for generating the benchmark synthetic data with an intrinsic community and hierarchical structure with influence
  on the network topology regulated by a scalar value.

## Command-line usage
To test the inference model on the given example file, type

```bash
python main.py
```
being inside the `src/scripts` folder. Default setting files contained in this repository are ready for **this command line test**.

It will use the sample network contained in `data/input`. The adjacency matrix *syn_test_r0.8_XOR_112.dat* represents a directed, weighted network with
**N=500** nodes, **K=3** equal-size unmixed communities with an **assortative** structure, **l=3** leagues and node type expected value **mu=0.8**.

### Parameters

- **-a** : Algorithm to use (MT, SR, XOR), *(default='XOR')*.
- **-K** : Number of communities, *(default=3)*.
- **-b** : Inverse temperature parameter, *(default=5)*.
- **-A** : Input file name of the adjacency matrix, *(default='syn_test_r0.8_XOR_112.dat')*.
- **-f** : Path of the input folder, *(default='../data/input/')*.
- **-F** : Number of folds to perform cross-validation, *(default=5)*.
- **-m** : Flag to output the cross validation masks, *(default=False)*.
- **-s** : Path of the settings file. If None, it is inferred using the selected algorithm, i.e. `../setting_<algorithm>.yaml`.

You can find a list by running (inside `src/scripts` directory):

```bash
python main.py --help
```

## Full demo

We provide a Jupyter Notebook [`XOR_demo.ipynb`](`XOR_demo.ipynb`) containing an easy-to-use interface with the XOR generative model and
the XOR inference model.
If not using Anaconda or another similar distribution, you will need to install `jupyter`:

```bash
pip install jupyter
```

## I/0 format

### Input

The network should be stored in a *.dat* file. An example  is

`source target w` <br>
`node1 node2 3` <br>
`node1 node3 1`

where the first and second columns are the _source_ and _target_ nodes of the edge, respectively; the third column tells  
the weight. In this example the edge node1 --> node2 exists with weight 3, and the edge node1 --> node3 exists with weight 1.

### Output
Using the configuration files as provided here, the algorithm returns a compressed file inside the `./data/output/XOR` folder.
To load and print the out-going membership matrix:

```py
import numpy as np
theta = np.load('path/to/output/folder/parameters_<experiment_label>.npz')
print(theta['u'])
```

_theta_ contains:
- the two NxK membership matrices **u** *('u')* and **v** *('v')*;
- the 1xKxK (or 1xK if assortative=True) affinity tensor **w** *('w')*;
- the model interaction coefficient **µ** *('mu')* (or, **ratio** *('ratio')*);
- the 1xN variational (or, a posteriori) probability tensor **Q** *('Q')*;
- the **δ<sub>0</sub>** *('delta0')* outgroup interaction coefficient if using the node-wise algorithm;
- the nodes of the network *('nodes_c')*, *('nodes_s')* reordered s.t. the induced adjacency tensor shows the block structure
  respectively given by the community and the hierarchical underlying models.  

Additional information on the experiment run can be found in the `metrics_<experiment_label>` label file in the same folder.
It can be loaded by using:

```py
import pandas as pd
metrics = pd.read_csv('path/to/output/folder/metrics_<experiment_label>.csv')
print(metrics.head())
```

_metrics_ contains:
- inference metrics values (if ground-truth available, `gt = True` in setting file):
  - L1 distance for u, v, w
  - average node-wise cosine similarity for u, v
  - Pearson and Spearman correlations for s, with their p-values
- metrics for node classification:  
  - AUC-ROC score (when `gt = True` in setting file)
  - AUC-PR score with respect class sigma = 1 (when `gt = True` in setting file)
  - AUC-PR score with respect class sigma = 0 (when `gt = True` in setting file)
  - AUC-ROC oracle score, computed using the ground-truth values of the latent variables (when `gt = True` in setting file)
- AUC-ROC score for edge prediction on training and test set (when `cv = True` in setting file)
- inferred values of the latent variables
- ground-truth values of the latent variables (when `gt = True` in setting file)
- hyperparameters values used for the run (K, beta, flag for constrained version)
- seed used for the run
- free energy value reached at the end of the run, denoted by `log_L`
- number of iterations performed
- flag for convergence
