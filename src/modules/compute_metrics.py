"""
    Functions for inference results processing.
"""


import re
import numpy as np
import pandas as pd

from numba import jit
from pathlib import Path
from sklearn import metrics
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr, poisson
from sklearn.metrics.pairwise import cosine_similarity as CS
try:
    from tools import import_data, delta_scores, load_npz_dict
except ModuleNotFoundError:
    from modules.tools import import_data, delta_scores, load_npz_dict

EPS = 1e-8

def Q_full(data, S, M, mu=None, delta0=None, Q=None, ratio=None, symmetric=False, **kwargs):
    """

        Computing a posteriori means for every type random variable (Z_ij or sigma_i),
        according to the XOR model.

        Parameters
        ---------
        data : ndarray
               Adjacency matrix.
        S : ndarray
            Matrix of all the SpringRank means Sij.
        M : ndarray
            Matrix of all the MultiTensor means Mij.
        mu : float
             A priori mean of Z/sigma, according to the XOR model.
        symmetric : bool
                    Flag for computing Q symmetric or asymmetric, in the XORe case.
        Returns
        ---------
        Q : float
            Z a posteriori mean.
    """

    if mu is None:
        if ratio is None:
            raise ValueError('No value for mu is passed!')
        else:
            mu = ratio

    # Compute probabilities of observing a weighted edge
    # given an expected value via poisson distribution
    AS = poisson.pmf(data, S)
    AM = poisson.pmf(data, M)

    # get number of nodes
    N = data.shape[-1]
    # augment dimensionality for numpy opt methods
    Qs_old = np.vstack([np.copy(Q)] * N)
    veclam = np.ones((N, N)) * delta0
    # compute probabilities of observing an edge
    # because of an outgroup interaction
    AL = poisson.pmf(data[0], veclam)
    # compute outer products
    ALt = np.einsum('ij,ji->ij', AL, AL)
    ASt = np.einsum('ij,ji->ij', AS[0], AS[0])
    AMt = np.einsum('ij,ji->ij', AM[0], AM[0])
    # logarithmic terms
    L1 = Qs_old * np.log(ASt + EPS) + (1. - Qs_old) * np.log(ALt + EPS)
    L2 = (1. - Qs_old) * np.log(AMt + EPS) + Qs_old * np.log(ALt + EPS)
    # j = i not influential on the final product: no self loops
    np.fill_diagonal(L1, 1.)
    np.fill_diagonal(L2, 1.)
    # compute exponents
    L1 = L1.sum(axis = 1) + np.log(mu + EPS)
    L2 = L2.sum(axis = 1) + np.log(1. - mu + EPS)
    phi1 = np.exp(L1)
    phi2 = np.exp(L2)
    max_phi = max(max(phi1), max(phi2))
    phi1 /= max_phi
    phi2 /= max_phi
    # Compute Q update using the formula:
    # phi1 = exp( sum_j[Q_j * log(AS_ij * AS_ji)] + sum_j[1-Q_j) * log(AL_ij * AL_ji)] )
    # phi2 = exp( sum_j[(1-Q_j) * log(AS_ij * AS_ji)] + sum_j[Q_j * log(AL_ij * AL_ji)] )
    Q_new = (phi1 / (phi1 + phi2))[np.newaxis, ...]

    return Q_new


def lambda_MT(u, v, w, **kargs):
    """
        Compute the mean M for all entries.

        Parameters
        ----------
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.

        Returns
        -------
        M : ndarray
    """

    if w.ndim == 2:
        M = np.einsum('ik,jk->ijk', u, v)
        M = np.einsum('ijk,ak->aij', M, w)
    else:
        M = np.einsum('ik,jq->ijkq', u, v)
        M = np.einsum('ijkq,akq->aij', M, w)

    return M


def lambda_SR(c, beta, s, l1=1., **kargs):
    """
        Compute the mean S for all entries.

        Parameters
        ----------
        c : float
            Sparsity coefficient.
        beta : float
               Inverse temperature.
        s : ndarray
            Ranking scores array.
        l1 : float
             Spring rest lenght.

        Returns
        -------
        S : ndarray
    """

    if s is not None:
        Ds = delta_scores(s.shape[0], s)
        return c * np.exp(-0.5 * beta * np.power(Ds - l1, 2))[np.newaxis, :, :]
    else:
        return 0


def lambda_XOR(c, beta, s, u, v, w, delta0, l1=1., S=None, M=None, Q=None, Q_old=None, data=None, **kargs):
    """
        Compute the a posteriori expectation matrix for the XOR model.

        Parameters
        ----------
        Q_old
        M
        S
        c : float
            Sparsity coefficient.
        beta : float
                Inverse temperature.
        s : ndarray
            Ranking scores array.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        Q : ndarray
             Z a posteriori mean.
        l1 : float
             Spring rest lenght.

        Returns
        -------
        lam : ndarray
    """

    if M is None:
        M = lambda_MT(u, v, w)
    if S is None:
        S = lambda_SR(c, beta, s, l1)
    N = M.shape[-1]
    vlam = np.ones((N, N)) * delta0

    if Q_old is not None:
        if data is not None:
            Q = Q_full(data, S, M, Q = Q_old, delta0 = delta0, **kargs)
        else:
            raise ValueError('Data needed for computing Q.')

    QQt = np.einsum('ai,aj->aij', Q, Q)
    Qs = np.vstack([Q] * N) + np.hstack([Q.T] * N)
    Qs = Qs[np.newaxis, ...]

    return QQt * (S + M - 2 * vlam) + Qs * (vlam - M) + M


def calculate_AUC(pred, data0, mask=None, curve='roc'):
    """
        Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
        (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
        (true negative).

        Parameters
        ----------
        pred : ndarray
               Inferred values.
        data0 : ndarray
                Given values.
        mask : ndarray
               Mask for selecting a subset of the adjacency matrix.
        curve : str
                Curve to be used.
        Returns
        -------
        AUC value.
    """
    if curve not in {'roc', 'prc'}:
        raise ValueError("Curve value must be 'roc' or 'prc'!")

    data = (data0 > 0).astype('int')
    if mask is None:
        if (data == 1).all() or (data == 0).all():
            return np.nan
        if curve == 'roc':
            x, y, thresholds = metrics.roc_curve(data.flatten(), pred.flatten())
        else:
            y, x, thresholds = metrics.precision_recall_curve(data.flatten(), pred.flatten())
    else:
        if (data[mask > 0] == 1).all() or (data[mask > 0] == 0).all():
            return np.nan
        if curve == 'roc':
            x, y, thresholds = metrics.roc_curve(data[mask > 0], pred[mask > 0])
        else:
            y, x, thresholds = metrics.precision_recall_curve(data[mask > 0], pred[mask > 0])
    return metrics.auc(x, y)


def normalize_nonzero_membership(u):
    """
    Given a matrix, it returns the same matrix normalized by row.

    Parameters
    ----------
    u: ndarray
    Numpy Matrix.

    Returns
    -------
    The matrix normalized by row.
    """

    den1 = u.sum(axis = 1, keepdims = True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1


@jit(nopython = True, parallel = True)
def opt_permutation(U_infer, U0):
    """
        Permuting the overlap matrix so that the groups from the two partitions correspond.

        Parameters
        ----------
        U_infer: ndarray
                Array of dimension NxK, inferred membership.
        U0 : ndarray
            Array of dimension NxK, reference membership.
        Returns
        -------
        P : ndarray
            Array of dimension KxK, optimal permutation.
    """

    N, K = U0.shape
    M = np.dot(np.transpose(U_infer), U0) / float(N)  # dim = KxK
    rows = np.zeros(K)
    columns = np.zeros(K)
    P = np.zeros((K, K))  # Permutation matrix
    for t in range(K):
        # Find the max element in the remaining submatrix,
        # the one with rows and columns removed from previous iterations
        max_entry = 0.
        c_index = 1
        r_index = 1
        for i in range(K):
            if columns[i] == 0:
                for j in range(K):
                    if rows[j] == 0:
                        if M[j, i] > max_entry:
                            max_entry = M[j, i]
                            c_index = i
                            r_index = j

        P[r_index, c_index] = 1
        columns[c_index] = 1
        rows[r_index] = 1

    return P


def cosine_similarity(v, v0):
    """
        Compute average cosine similarity on matrices.
    """
    cosine_sim = CS(v, v0).diagonal()
    return cosine_sim.mean()


def membership_metrics(v, v0, eps=1e-8):
    """
        Compute average L1 and cosine distance metrics for the membership vectors.

        Parameters
        ----------
        v: ndarray
            Array of dimension NxK, inferred membership row-normalized.
        v0 : ndarray
             Array of dimension NxK, reference membership row-normalized.
        eps : float
              Approximation of zero.
        Returns
        -------
        L1 : float
             Average L1 error.
        CD : float
             Average cosine distance ( 1 - cosine similarity ).
    """

    P = opt_permutation(v, v0)
    v = np.dot(v, P)  # permute on clusters (2nd dimension)
    L1 = 1 / 2 / v.shape[0] * np.linalg.norm((v - v0), ord = 1)
    v[np.sum(v, axis = 1) == 0.] = np.ones_like(v[0]) * eps  # avoid NaN CD
    CS = cosine_similarity(v, v0)
    CD = 1. - CS
    return L1, CD


def rmetrics(out, lab, verbose=1):
    """
        Wrapper for computing all the ranking related metrics. Same input as metrics.
        Returns, in order: pearson correlation, pearson p-value, spearman correlation,
        spearman p-value.
    """
    nodes = np.argsort(lab['s'])[::-1]
    try:
        s, ns = out['s'], out['nodes_s']
        pcorr = pearsonr(s, lab['s'])
        scorr = spearmanr(ns, nodes)
        if verbose:
            print(f'Pearson correlation on positions: {pcorr[0]}, p-value: {pcorr[1]} ')
            print(f'Spearman correlation on ranks: {scorr[0]}, p-value: {scorr[1]} ')
            print('\nInferred:\n', out['nodes_s'])
            print('\nLabels:\n', nodes)
        return [pcorr[0], pcorr[1], scorr[0], scorr[1]]
    except:
        return [np.nan] * 4


def cmetrics(out, lab, verbose=1):
    """
        Wrapper for computing all the community detection related metrics. Same input as metrics.
        Returns, in order: L1u, CDu, L1v, CDv.
    """
    try:
        u = normalize_nonzero_membership(out['u'])
        v = normalize_nonzero_membership(out['v'])
        L1u, CDu = membership_metrics(u, lab['u'])
        L1v, CDv = membership_metrics(v, lab['v'])
        if verbose:
            print('L1 norm on u:', L1u)
            print('Cosine distance on u:', CDu)
            print('L1 norm on v:', L1v)
            print('Cosine distance on v:', CDv, end = '\n\n')
        return [L1u, CDu, L1v, CDv]
    except KeyError:
        return [None] * 4


def edge_metrics(data, M_lab, S_lab, mask, model, out, lab):
    """
        Wrapper for computing all the edge prediction related metrics.
        Returns the AUC score on train and test sets.
    """

    if model == 'XOR':
        expectation = lambda_XOR
    elif model == 'MT':
        expectation = lambda_MT
    elif model == 'SR':
        expectation = lambda_SR

    S_out = None; M_out = None
    # compute edge type probabilities
    if model != 'MT': S_out = lambda_SR(**out)
    if model != 'SR': M_out = lambda_MT(**out)

    pred = expectation(**out, S=S_out, M=M_out)

    assert np.isfinite(pred).all()

    AUC_train = calculate_AUC(pred, data, mask=np.logical_not(mask))
    AUC_test = calculate_AUC(pred, data, mask=mask)

    if M_lab is not None:
        Z_lab = lab.get('sigma')
        if Z_lab is None:
            Z_lab = lab.get('z')
        try:
            lab['Q'] = Z_lab[np.newaxis, ...]
            pred = expectation(**lab, S=S_lab, M=M_lab)
            oracle_train = calculate_AUC(pred, data, mask=np.logical_not(mask))
            oracle_test = calculate_AUC(pred, data, mask=mask)
            return [AUC_train, AUC_test, oracle_train, oracle_test]
        except TypeError:
            return [AUC_train, AUC_test, None, None]
    else:
        return [AUC_train, AUC_test, None, None]


def class_metrics(Z, Q_out, Q_lab, mask=None, oracle=True):
    """
        Wrapper for computing all the classification related metrics.
        Returns the AUC score and the oracle value.
    """

    AUC_PRSR = calculate_AUC(Q_out, Z, mask = mask, curve = 'prc')
    AUC_PRMT = calculate_AUC(1 - Q_out, 1 - Z, mask = mask, curve = 'prc')
    AUC_ratio = calculate_AUC(Q_out, Z, mask = mask)
    if oracle:
        oracle_ratio = calculate_AUC(Q_lab, Z, mask = mask)
        return [AUC_PRSR, AUC_PRMT, AUC_ratio, oracle_ratio]
    return [AUC_PRSR, AUC_PRMT, AUC_ratio, None]


def inference_metrics(out, lab, model, data, M_lab, S_lab, verbose=1, classification=False):
    """
        Wrapper for computing all the latent variable related metrics plus printing information.

        Parameters
        ----------
        out : dict
              Dict with inferred values.
        lab : dict
              Dict with ground truth values.
        model : str
                Model label.
        data : ndarray
               Adjacency tensor.
        M_lab : ndarray
                Tensor of MT means computed with GR latent variables.
        S_lab : ndarray
                Tensor of SR means computed with GR latent variables.
        verbose : int
                  Verbosity value for the info printing, can be 0 or 1.
        classification: bool
                        Flag for computing and saving the classification metrics.
        Returns
        -------
        met : list
             List with all the metric values.
        cols : list
             List with all the metric labels, in order.
    """

    cmet = cmetrics(out, lab, verbose)
    rmet = rmetrics(out, lab, verbose)
    # unify metrics
    met = cmet + rmet
    # create list of column names ist
    cols = ['L1u', 'CDu', 'L1v', 'CDv', 'PearsonCoeff', 'PearsonPvalue', 'SpearmanCoeff', 'SpearmanPvalue']
    if classification and model == 'XOR': # avoid to try computing this when model in {SR, MT}
        # compute edge type probabilities
        S_out = lambda_SR(**out)
        M_out = lambda_MT(**out)
        Q_out = Q_full(data, S_out, M_out, **out)
        # get GT latent variable
        Z_lab = lab.get('sigma')
        if Z_lab is None:
            Z_lab = lab.get('z')
        Z_lab = Z_lab[np.newaxis, ...]
        try:
            oracle = True
            Q_lab = Q_full(data, S_lab, M_lab, Q = Z_lab, **lab)
        except TypeError:
            oracle = False
            Q_lab = None
        # get metrics
        AUC_PRSR, AUC_PRMT, AUC_ratio, oracle_ratio = class_metrics(Z_lab, Q_out, Q_lab, oracle = oracle)
        cols.extend(['AUC_PRSR', 'AUC_PRMT', 'AUC_ratio', 'oracle_ratio'])
        met.extend([AUC_PRSR, AUC_PRMT, AUC_ratio, oracle_ratio])

    return met, cols


def save_metrics(out, label_path, out_path, model='XOR', mask=None, clas=False, cv=False, ground_truth=False):
    """
        Compute and save to .csv the metrics for the current experiment realization.
        Realizations for the same output are saved in a unique output file.

        Parameters
        --------
        out: dict
            Dictionary containing all the results from the inference algorithm.
        label_path: string
                    Path for loading the ground truth parameters.
        out_path: string
                  Path for saving the metrics file.
        clas: bool
              Flag for computing and saving the classification metrics.
        cv: bool
            Flag for computing and saving the CV metrics.
        ground_truth: bool
                      Flag for computing and saving the inference metrics.
    """

    mlist = []
    cols = []
    data_path = None
    M_lab = None
    S_lab = None
    lab = {}

    if ground_truth:
        # load dict of ground truth values
        lab = load_npz_dict(label_path + '.npz')
        lab['w'] = lab['w'][np.newaxis, :, :]
        # ground truth means
        M_lab = lambda_MT(lab['u'], lab['v'], lab['w'])
        try:
            S_lab = lambda_SR(lab['c'], out['beta'], lab['s'])
        except KeyError:
            S_lab = lambda_SR(lab['c'], lab['beta'], lab['s'])
        # load data
        data_path = re.sub('results', 'syn', label_path) + '.dat'
        data = import_data(data_path, header = 0, force_dense = True, verbose = 0)[1]
        # compute inference metrics (latent variables + edge/node classification)
        m, c = inference_metrics(out, lab, model, data, M_lab, S_lab,  verbose = 0, classification = clas)
        mlist.extend(m)
        cols.extend(c)
    if cv:
        if data_path is None:
            # load data
            data_path = re.sub('results_', '', label_path) + '.dat'
            data = import_data(data_path, header = 0, force_dense = True, verbose = 0)[1]
        m = edge_metrics(data, M_lab, S_lab, mask, model, out, lab)
        c = ['AUC_train', 'AUC_test', 'oracle_train', 'oracle_test']
        mlist.extend(m)
        cols.extend(c)
    # add output and label values to the list - excluding vectorial variables
    _, _, _ = out.pop('Q', 0), out.pop('nodes_s', 0), out.pop('nodes_c', 0)
    _, _, _, _, _ = lab.pop('Q', 0), lab.pop('z', 0), lab.pop('edge_mask', 0), lab.pop('nodes', 0), lab.pop('nodes_s', 0)
    mlist.extend(list(out.values()))
    mlist.extend(list(lab.values()) if lab is not None else [])
    # add output and label names to the list - excluding vectorial variables
    cols.extend([x + '_out' for x in out.keys()])
    cols.extend([x + '_lab' for x in lab.keys()] if lab is not None else [])

    file = Path(out_path + '.csv')
    if not file.is_file():
        df = pd.DataFrame(data = [mlist], columns = cols)
    else:
        df = pd.read_csv(out_path + '.csv', engine = 'python', error_bad_lines = False)
        new_row = pd.DataFrame(data = [mlist], columns = cols, dtype = object)
        df = pd.concat([df, new_row], ignore_index = True)
    df.to_csv(out_path + '.csv', index = False)
