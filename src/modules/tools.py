"""
    Functions for handling the data before and after inference.
"""

import numpy as np
import pandas as pd
import networkx as nx
import sktensor as skt

from numba import jit
from sklearn.preprocessing import minmax_scale

'''   (1) Graph handling functions   '''

def load_npz_dict(path):
    """
        Load the npz file as a dictionary.

        Parameters
        ----------
        path : string
               Full path to .npz file.
        Returns
        --------
        saved_dict : dict
                A dictionary of parameters.
    """
    d = np.load(path, allow_pickle = True)
    saved_dict = {k: v for k, v in d.items()}
    return saved_dict


def import_data(dataset, ego='source', alter='target', force_dense=True, header=None, noselfloop=True, verbose=1):
    """
        Import data, i.e. the adjacency matrix, from a given folder.

        Return the NetworkX graph and its numpy adjacency matrix.

        Parameters
        ----------
        dataset : str
                  Path of the input file.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        force_dense : bool
                      If set to True, the algorithm is forced to consider a dense adjacency tensor.
        header : int
                 Row number to use as the column names, and the start of the data.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.
        verbose : int or bool
                  Flag for function verbosity, level 0 (False) or 1 (True).
        Returns
        -------
        A : list
            List of DiGraph NetworkX objects.
        B : ndarray/sptensor
            Graph adjacency tensor.
    """

    # read adjacency file
    df_adj = pd.read_csv(dataset, sep = '\s+', header = header)
    if verbose:
        print('{0} shape: {1}'.format(dataset, df_adj.shape))

    A = read_graph(df_adj = df_adj, ego = ego, alter = alter)

    nodes = list(A[0].nodes())

    # save the network in a tensor
    if force_dense:
        B = build_B_from_A(A, nodes = nodes)
    else:
        B = build_sparse_B_from_A(A)

    if verbose:
        print_graph_stat(A)

    return A, B


def read_graph(df_adj, ego='source', alter='target', noselfloop=True):
    """
        Create the graph by adding edges and nodes.
        It assumes that columns of layers are from l+2 (included) onwards.
        Return the list MultiDiGraph NetworkX objects.
        Parameters
        ----------
        df_adj : DataFrame
                 Pandas DataFrame object containing the edges of the graph.
        ego : str
              Name of the column to consider as source of the edge.
        alter : str
                Name of the column to consider as target of the edge.
        noselfloop : bool
                     If set to True, the algorithm removes the self-loops.
        Returns
        -------
        A : list
            List of MultiDiGraph NetworkX objects.
    """

    # build nodes
    egoID = df_adj[ego].unique()
    alterID = df_adj[alter].unique()
    nodes = list(set(egoID).union(set(alterID)))
    nodes.sort()

    L = df_adj.shape[1] - 2  # number of layers
    # build the NetworkX graph: create a list of graphs, as many graphs as there are layers
    A = [nx.MultiDiGraph() for _ in range(L)]
    # set the same set of nodes and order over all layers
    for l in range(L):
        A[l].add_nodes_from(nodes)

    for index, row in df_adj.iterrows():
        v1 = row[ego]
        v2 = row[alter]
        for l in range(L):
            if row[l + 2] > 0:
                if A[l].has_edge(v1, v2):
                    A[l][v1][v2][0]['weight'] += int(row[l + 2])  # the edge already exists -> no parallel edge created
                else:
                    A[l].add_edge(v1, v2, weight=int(row[l + 2]))

    # remove self-loops
    if noselfloop:
        for l in range(L):
            A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

    return A


def print_graph_stat(A):
    """
        Print the statistics of the graph A.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
    """

    L = len(A)
    N = A[0].number_of_nodes()
    print('Number of nodes =', N)
    print('Number of layers =', L)

    print('Number of edges and average degree in each layer:')
    for l in range(L):
        E = A[l].number_of_edges()
        k = float(E) / float(N)
        M = np.sum([d['weight'] for _, _, d in list(A[l].edges(data = True))])
        kW = float(M) / float(N)

        print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')
        print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')


def build_B_from_A(A, nodes=None):
    """
        Create the numpy adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
        nodes : list
                List of nodes IDs.

        Returns
        -------
        B : ndarray
            Graph adjacency tensor.
    """

    N = A[0].number_of_nodes()
    if nodes is None:
        nodes = list(A[0].nodes())
    B = np.empty(shape = [len(A), N, N])
    for l in range(len(A)):
        B[l, :, :] = nx.to_numpy_array(A[l], weight = 'weight', dtype = int, nodelist = nodes)
    return B


def build_sparse_B_from_A(A):
    """
        Create the sptensor adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.

        Returns
        -------
        data : sptensor
               Graph adjacency tensor.
    """

    N = A[0].number_of_nodes()
    L = len(A)

    d1 = np.array((), dtype = 'int64')
    d2 = np.array((), dtype = 'int64')
    d3 = np.array((), dtype = 'int64')
    v = np.array(())
    for l in range(L):
        b = nx.to_scipy_sparse_matrix(A[l])
        nz = b.nonzero()
        d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
        d2 = np.hstack((d2, nz[0]))
        d3 = np.hstack((d3, nz[1]))
        v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
    subs_ = (d1, d2, d3)
    data = skt.sptensor(subs_, v, shape = (L, N, N), dtype = v.dtype)

    return data


'''   (2) CV and inference functions    '''

@jit(nopython = True, parallel = True)
def delta_scores(n, s):
    """
        Compute the pairwise ranking differences.
    """

    delta_s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            delta_s[i, j] = s[i] - s[j]
    return delta_s


def shuffle_indices_all_matrix(N, L, rseed=10):
    """
        Shuffle the indices of the adjacency matrix.

        Parameters
        ----------
        N : int
            Number of nodes.
        L : int
            Number of layers.
        rseed : int
                Random seed.

        Returns
        -------
        indices : ndarray
                  Indices in a shuffled order.
    """

    n_samples = int(N * N)
    indices = [np.arange(n_samples) for _ in range(L)]
    rng = np.random.RandomState(rseed)
    for l in range(L):
        rng.shuffle(indices[l])

    return indices


def extract_mask_kfold(indices, N, fold=0, NFold=5):
    """
        Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but possibly not (j,i).
        KFold means no train/test sets intersect across the K folds.

        Parameters
        ----------
        indices : ndarray
                  Indices of the adjacency matrix in a shuffled order.
        N : int
            Number of nodes.
        fold : int
               Current fold.
        NFold : int
                Number of total folds.

        Returns
        -------
        mask : ndarray
               Mask for selecting the held out set in the adjacency matrix.
    """

    L = len(indices)
    mask = np.zeros((L, N, N), dtype = bool)
    for l in range(L):
        n_samples = len(indices[l])
        test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
        mask0 = np.zeros(n_samples, dtype = bool)
        mask0[test] = 1
        mask[l] = mask0.reshape((N, N))

    return mask
