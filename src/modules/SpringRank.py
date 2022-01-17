"""
    Class definition of SpringRank.
"""


import sparse
import warnings
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from scipy.optimize import brentq
from compute_metrics import save_metrics

from tools import delta_scores

class SpringRank(object):

    def __init__(self, N=100, L=1, gamma=0., l0=1., l1=1., solver='bicgstab', a=0.01, b=20.,
                 inf=1e10, verbose=0, force_dense=False, shift_rank=False, get_beta=True,
                 out_inference=False, out_folder='../data/output/', in_folder=None, label='',
                 cv=False, gt=False, **kargs):

        self.N = N  # number of nodes
        self.L = L  # number of layers
        self.gamma = gamma  # regularization penalty - spring constant for the fictitious i <-> origin connections
        self.l0 = l0  # resting length for the fictitious i <-> origin connections
        self.l1 = l1  # resting length for the i <-> j connections
        self.inf = inf  # infinity value
        self.force_dense = force_dense  # flag for forcing the algorithm to use dense matrices
        self.shift_rank = shift_rank  # Flag for shifting ranks to positive
        self.get_beta = get_beta  # flag for inferring the inverse temperature parameter
        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.in_folder = in_folder  # path for reading the labels
        self.label = label  # additional label for the output
        self.cv = cv  # flag for including cv metrics in the output
        self.gt = gt  # flat for including metrics wrt ground truth in the output

        if solver not in {'spsolve', 'bicgstab'}:  # solver used for the linear system
            warnings.warn(f'Unknown parameter {solver} for argument solver. Setting solver = "bicgstab"')
            solver = 'bicgstab'
        self.solver = solver
        if a < 0 or a > b:  # beta search interval
            raise ValueError(
                'The [a,b] interval for beta is not valid! a must be positive and b must be greater then a.')
        self.a = a
        self.b = b
        if verbose > 2 and not isinstance(verbose, int):  # verbosity indicator
            raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
        self.verbose = verbose

        if self.verbose == 2:
            print(f'Using scipy.sparse.linalg.{self.solver}(A,B)')

    def fit(self, data, mask=None):
        """
            Model directed networks by using assuming the existence of a hierarchical structure.
            The ranking scores (unidimensional embeddings) are inferred by solving a linear system.

            Parameters
            ----------
            data : ndarray/spmatrix
                   Has tobe  2 dimensional and with same dimensions.

            Returns
            -------
            rank : ndarray
                   Array of ranks. Indices represent the nodes' indices used in the input matrix.
        """

        if self.L > 2:
            raise NotImplementedError('SpringRank for tensors not implemented! Use 2-dimensional input.')

        if len(data.shape) > 2:
            data = data[0]
        if mask is not None and len(mask.shape) > 2:
            mask = mask[0]

        # check if input is sparse or can be converted to sparse.
        use_sparse = True
        if not self.force_dense and not scipy.sparse.issparse(data):
            try:
                data = scipy.sparse.csr_matrix(data)
            except:
                warnings.warn('The input parameter A could not be converted to scipy.sparse.csr_matrix. '
                              'Using a dense representation.')
                use_sparse = False
        elif self.force_dense:
            use_sparse = False

        # build array to feed linear system solver
        if use_sparse:
            A, B = self._build_from_sparse(data)
        else:
            A, B = self._build_from_dense(data)

        rank = self._solve_linear_system(A, B)

        if self.shift_rank:
            rank = shift_rank(rank)
        self.s = rank

        self.beta, self.c = None, None
        if self.get_beta:
            self.beta = self._get_optimal_temperature(data)
            self.c = self._get_sparsity_coefficient(data)

        if self.out_inference:
            self._output_results(mask = mask)

        return self.s, self.beta, self.c

    def _build_from_dense(self, data):
        """
        Given as input a 2d numpy array, build the matrices A and B to feed to the linear system solver for SpringRank.
        """

        k_in = np.sum(data, 0)
        k_out = np.sum(data, 1)

        D1 = k_in + k_out  # to be seen as diagonal matrix, stored as 1d array
        D2 = self.l1 * (k_out - k_in)  # to be seen as diagonal matrix, stored as 1d array

        if self.gamma != 0.:
            B = np.ones(self.N) * (self.gamma * self.l0) + D2
            A = - (data + data.T)
            A[np.arange(self.N), np.arange(self.N)] = self.gamma + D1 + np.diagonal(A)
        else:
            last_row_plus_col = (data[n - 1, :] + data[:, n - 1]).reshape((1, self.N))
            A = data + data.T
            A += last_row_plus_col

            A[np.arange(self.N), np.arange(self.N)] = A.diagonal() + D1
            D3 = np.ones(self.N) * (
                        self.l1 * (k_out[n - 1] - k_in[n - 1]))  # to be seen as diagonal matrix, stored as 1d array
            B = D2 + D3

        return scipy.sparse.csr_matrix(A), B

    def _build_from_sparse(self, data):
        """
        Given as input a sparse 2d scipy array, build the matrices A and B to feed to the linear system solver for
        SpringRank.
        """

        k_in = np.sum(data, 0).A1  # convert matrix of shape (1, n) into 1-dimensional array
        k_out = np.sum(data, 1).A1  # same with (n, 1) matrix

        D1 = k_in + k_out  # to be seen as diagonal matrix, stored as 1d array
        D2 = self.l1 * (k_out - k_in)  # to be seen as diagonal matrix, stored as 1d array

        if self.gamma != 0.:
            B = np.ones(self.N) * (self.gamma * self.l0) + D2
            A = - (data + data.T)
            # convert to lil matrix for more efficient computations
            A = A.tolil(copy = False)
            A.setdiag(self.gamma + D1 + A.diagonal())
        else:
            last_row_plus_col = sparse.COO.from_scipy_sparse(
                data[self.N - 1, :] + data[:, self.N - 1].T)  # create sparse 1d COO array
            A = data + data.T
            A += last_row_plus_col  # broadcast on rows
            A = -A.tocsr()  # reconvert to csr scipy matrix

            # Notice that a scipy.sparse.SparseEfficiencyWarning will be raised by calling A.setdiag().
            # However converting to lil matrix with
            # A.tolil(copy=False)
            # is not computationally convenient. Just suppress the warning during the call of A.setdiag(...)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.sparse.SparseEfficiencyWarning)
                A.setdiag(A.diagonal() + D1)

            D3 = np.ones(self.N) * (self.l1 * (
                        k_out[self.N - 1] - k_in[self.N - 1]))  # to be seen as diagonal matrix, stored as 1d array
            B = D2 + D3
        return A, B

    def _solve_linear_system(self, A, B):

        if self.solver == 'spsolve':
            sol = scipy.sparse.linalg.spsolve(A, B)
        elif self.solver == 'bicgstab':
            sol = scipy.sparse.linalg.bicgstab(A, B, tol = 1e-08, atol = 'legacy')[0]
        return sol.reshape((-1,))

    def _get_optimal_temperature(self, A):
        if eq_beta(self.a, self.s, A) * eq_beta(self.b, self.s, A) > 0:
            if self.verbose == 2:
                print(f'Beta update computed in the interval [0,+inf) instead of [0,{self.b}].')
            self.b = self.inf
        return brentq(eq_beta, self.a, self.b, args = (self.s, A))

    def _get_sparsity_coefficient(self, A):
        A = A.todense()
        H = - 0.5 * np.power(delta_scores(A.shape[0], self.s), 2) * self.beta
        return A[A != 0].sum() / np.exp(H)[A != 0].sum()

    def _output_results(self, mask=None):
        """
            Output results in a compressed file.
            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
        """

        # saving s and nodes sorted by s values
        nodes = np.argsort(self.s)[::-1]
        output_parameters = self.out_folder + 'parameters_' + self.label + '_SR'

        np.savez_compressed(output_parameters + '.npz', s = self.s, beta = self.beta, c = self.c, nodes = nodes)
        if self.in_folder is not None:
            out = {'s': self.s, 'beta': self.beta, 'c': self.c, 'gamma': self.gamma, 'nodes_s': nodes}
            out_metrics = self.out_folder + 'metrics_' + self.label + '_SR'
            label_path = self.in_folder
            save_metrics(out, label_path, out_metrics, model = 'SR', mask = np.logical_not(mask)[None, :, :] if mask is not None else None,
                         cv = self.cv, ground_truth = self.gt)

        if self.verbose == 2:
            print()
            print(f'Parameters saved in: {output_parameters}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]')
            if self.in_folder is not None:
                print(f'Metrics saved in: {out_metrics}.csv')
                print('Load as a pandas dataframe.', end = '\n\n')


def eq_beta(beta, s, A):
    # optimal beta wrt conditional likelihood (eq.S39)
    N = np.shape(A)[0]
    x = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                continue
            else:
                x += (s[i] - s[j]) * (A[i, j] - (A[i, j] + A[j, i]) / (1 + np.exp(-2 * beta * (s[i] - s[j]))))
    return x


def shift_rank(ranks):
    """
        Shifts all scores by translations, so that the minimum is in zero
        and the others are all positive
    """

    min_r = min(ranks)
    N = len(ranks)
    for i in range(N):
        ranks[i] = ranks[i] - min_r
    return ranks
