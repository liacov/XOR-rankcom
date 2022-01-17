"""
    Main function to run the XOR inference algorithm on a given single-layer directed network. The subcases SpringRank
    and MultiTensor are included for baseline comparison.

    Is it possible to use cross-validation, i.e.
        - Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
        - Infer parameters on the training set;
        - Calculate performance measures in the test set.
    and grid search for optimizing the number of communities and the beta parameter (inverse temperature parameter for
    the hierarchical structure inference task).
"""

import sys

sys.path.append('../modules')

import os
import re
import yaml
import time
import numpy as np
from argparse import ArgumentParser
from tools import import_data, shuffle_indices_all_matrix, extract_mask_kfold

import XOR as XOR
import SpringRank as SR
import MultiTensor as MT


def main(args):

    # setting to run the algorithm
    if args['settings'] is None:
        settings_path = '../setting_' + args['algorithm'] + '.yaml'
    else:
        settings_path = args['settings']
    with open(settings_path) as f:
        conf = yaml.load(f, Loader = yaml.FullLoader)
    # create output folder
    if not os.path.exists(conf['out_folder']):
        os.makedirs(conf['out_folder'], exist_ok = True)

    # set data label depending on data generation type
    if conf['gt']:
        conf['label'] = re.sub('syn_', '', args['adj'][:-4])  # synthetic data
    else:
        conf['label'] = args['adj'][:-4]  # real data

    '''
    Model parameters
    '''
    network = args['in_folder'] + args['adj']  # network complete path
    algorithm = args['algorithm']  # algorithm to use to generate the samples

    '''
    Import data
    '''
    A, B = import_data(network, header = 0, force_dense = True)
    nodes = A[0].nodes()
    # reconstruct path to corresponding ground truth
    conf['in_folder'] = args['in_folder'] + 'results_' + conf['label']

    assert isinstance(B, np.ndarray)

    '''
    Run main routine for the chosen inference model.
    '''
    if algorithm in {'SR', 'MT'}:
        conf['classification'] = False

    if conf['cv']:
        # create subfolder for CV
        conf['out_folder'] += '{}-fold_cv/'.format(args['NFold'])
        if not os.path.exists(conf['out_folder']):
            os.makedirs(conf['out_folder'], exist_ok = True)

        # name of the network without extension
        adjacency = args['adj'].split('.dat')[0]
        # run inference
        run_inference_cv(B, nodes, algorithm, adjacency, args, conf)
    else:
        # run inference
        run_inference(B, nodes, algorithm, args, conf)


def run_inference(B, nodes, algorithm, args, conf):
    """

    Parameters
    ----------
    B
    nodes
    algorithm
    args
    conf

    """

    # Get dimensions
    L = B.shape[0]
    N = B.shape[-1]
    K = args['K']
    beta0 = args['beta']

    if isinstance(K, list) and len(K) == 1:
        K = K[0]
    elif isinstance(K, int):
        pass
    else:
        raise ValueError('For testing the algorithm with different number of communities K, please use the Cross '
                         'Validation option (yalm configuration file). ')

    print(f'\n--- Run inference with {algorithm} model ---', end = '\n\n')

    time_start = time.time()
    if algorithm == 'XOR':
        model = XOR.EitherOr(N = N, L = L, K = K, beta0 = beta0, **conf)
    elif algorithm == 'SR':
        model = SR.SpringRank(N = N, L = L, **conf)
    elif algorithm == 'MT':
        model = MT.MultiTensor(N = N, L = L, K = K, **conf)
    else:
        raise ValueError('Selected model not in the list of available models. Please use SR, MT or XOR!')

    # Run inference
    if algorithm != 'SR':
        out = model.fit(data = B, nodes = nodes)
        if conf['out_inference']:
            # Output latent variables and stats
            model.save_results(out, **conf)
    else:
        _ = model.fit(data = B)

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


def run_inference_cv(B, nodes, algorithm, adjacency, args, conf):
    """

    Parameters
    ----------
    B
    nodes
    algorithm
    adjacency
    args
    conf

    """
    # Set seed random number generator
    prng = np.random.RandomState(seed = 42)
    # Get dimensions
    L = B.shape[0]
    N = B.shape[-1]
    K = args['K']

    if not isinstance(K, list):
        K = [K]

    print(f'\n---  Run inference with CV procedure, {algorithm} model ---')

    time_start = time.time()

    rseed = prng.randint(1000)
    indices = shuffle_indices_all_matrix(N, L, rseed = rseed)
    init_end_file = conf['label']

    for fold in range(args['NFold']):
        print('\nFOLD ', fold)

        mask = extract_mask_kfold(indices, N, fold = fold, NFold = args['NFold'])
        if args['out_mask']:
            outmask = conf['out_folder'] + 'mask_f' + str(fold) + '_' + adjacency + '_inf' + algorithm + '.npz'
            np.savez_compressed(outmask, mask = mask)
            print(f'Mask saved at: {outmask}')

        '''
        Set up training dataset
        '''
        B_train = B.copy()
        B_train[mask > 0] = 0

        '''
        Run on the training for every K
        '''
        tic = time.time()

        conf['label'] = init_end_file + '_' + str(fold)
        fit_cv(B_train, nodes, N, L, K, args['beta'], algorithm, mask = np.logical_not(mask), **conf)

        print(f'Time elapsed for fold {fold}: {np.round(time.time() - tic, 2)} seconds.')

    print(f'Total time elapsed: {np.round(time.time() - time_start, 2)} seconds.')


def fit_cv(B, nodes, N, L, K, b, alg, mask=None, **conf):
    """
        Model directed networks by using a probabilistic generative model that assume community
        and/or hierarchical structure. The inference is performed via EM algorithm.

        Parameters
        ----------
        B : ndarray
            Graph adjacency tensor.
        nodes : list
                List of nodes IDs.
        N : int
            Number of nodes.
        L : int
            Number of layers.
        K : list int
            List of number of communities.
        b: float
           Inverse temperature parameter.
        alg : str
                    Configuration to use (MM, XOR, SR, MT).
        mask : ndarray
               Mask for cv.

    """

    label = conf['label']

    if alg == 'SR':
        model = SR.SpringRank(N = N, L = L, **conf)
        _ = model.fit(data = B, mask = mask)

    else:
        for k in K:
            print(f'Using K = {k}.')
            conf['label'] = label + f'_K{k}'
            if alg == 'MT':
                model = MT.MultiTensor(N = N, L = L, K = k, **conf)
            elif alg == 'XOR':
                model = XOR.EitherOr(N = N, L = L, K = k, beta0 = b, **conf)
            else:
                raise ValueError('Selected model not in the list of available models. Please use SR, MT or XOR!')

            # Run inference
            out = model.fit(data = B, nodes = nodes, mask = mask)
            if conf['out_inference']:
                # Output latent variables and stats
                model.save_results(out, mask = mask, **conf)
            conf['rseed'] += 100000


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('-a', '--algorithm', type = str, choices = ['XOR', 'SR', 'MT'], default = 'XOR',
                   help = 'Code for the desired model, among XOR, SpringRank, MultiTensor. '
                          'Default = "XOR"')
    p.add_argument('-K', '--K', type = int, nargs = '+', default = 3, help = 'Number of communities to infer.')
    p.add_argument('-b', '--beta', type = float, default = 5, help = 'Inverse temperature parameter.')
    p.add_argument('-A', '--adj', type = str, default = 'syn_test_r0.8_XOR_112.dat', help = 'Name of the network file.')
    p.add_argument('-f', '--in_folder', type = str, default = '../../data/input/', help = 'Folder path of the input '
                                                                                          'network.')
    p.add_argument('-F', '--NFold', type = int, default = 5, help = 'Number of folds to perform cross-validation')
    p.add_argument('-m', '--out_mask', type = bool, default = False, help = 'Flag to output the masks')
    p.add_argument('-s', '--settings', type = str, help = 'Path of the settings file. If None, it is inferred using the'
                                                          ' selected algorithm, i.e. "../setting_<algorithm>.yaml"')
    args = p.parse_args()

    main(vars(args))
