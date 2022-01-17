"""
    Class definition of XOR, the algorithm to perform inference in networks assuming a mixed effect of the community
    and hierarchical latent structures.
"""

from __future__ import print_function

import sys
import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse
import sktensor as skt
import SpringRank as SR
import MultiTensor as MT

from termcolor import colored
from tools import delta_scores
from scipy.stats import poisson, entropy
from compute_metrics import save_metrics

EPS = 1e-8

# noinspection PyAttributeOutsideInit
class EitherOr(MT.MultiTensor):

    def __init__(self, N=100, L=1, K=2, initialization=0, rseed=42, inf=1e10, err_max=1e-8, err=0.01, N_real=1,
                 tolerance=0.001, decision=5, max_iter=500, out_inference=False, out_folder='../data/output/',
                 in_folder=None, label='', assortative=False, verbose=0, fix_mu=False, fix_scores=False,
                 fix_communities=False, fix_means=False, fix_delta=False, beta0=None, c0=1., mu0=0.5, delta0=0.001,
                 solver='bicgstab', gamma=0., constrained=False, l0=0., l1=1., classification=True, randomize_mu=True,
                 lambda_u=5., lambda_v=5., lambda_w=10., cv=False, gt=False, input_s='../data/input/s.dat',
                 input_u='../data/input/u.dat', input_v='../data/input/v.dat',
                 input_w='../data/input/w.dat', input_Q='../data/input/sigma.dat'):

        # ---- Attributes shared with MultiTensor  ----
        super().__init__(N = N, L = L, K = K, initialization = initialization, rseed = rseed, inf = inf, err = err,
                         err_max = err_max, N_real = N_real, tolerance = tolerance, decision = decision,
                         max_iter = max_iter, out_inference = out_inference, label = label, out_folder = out_folder,
                         in_folder = in_folder, assortative = assortative, verbose = verbose, input_u = input_u,
                         input_v = input_v, input_w = input_w, constrained = constrained, lambda_u = lambda_u,
                         lambda_v = lambda_v, lambda_w = lambda_w, cv = cv, gt = gt)
        # ---- XOR-specific attributes ----
        self.input_s = input_s  # path of the input file s (when initialization=1)
        self.input_Q = input_Q  # path of the input file s (when initialization=1)
        self.fix_scores = fix_scores  # flag for fixing ranking latent variable s to ground truth values
        self.fix_communities = fix_communities  # flag for fixing community latent variables to ground truth values
        self.fix_means = fix_means  # flag for fixing the prior and posterior mean of sigma to ground truth value
        self.fix_mu = fix_mu  # flag for fixing the prior mean of sigma to ground truth value
        self.fix_delta = fix_delta  # flag for fixing the outgroup interaction mean delta_0 to ground truth value
        self.beta = beta0  # initial value for the inverse temperature
        self.gamma = gamma  # regularization penalty - spring constant for the fictitious i <-> origin connections
        self.l0 = l0  # resting length for the fictitious i <-> origin connections
        self.l1 = l1  # resting length for the i <-> j connections
        self.classification = classification  # flag for computing classification metrics
        self.randomize_mu = randomize_mu  # flag for randomly generating mu
        if solver not in {'spsolve', 'bicgstab'}:  # solver used for the SR linear system
            warnings.warn(f'Unknown parameter {solver} for argument solver. Setting solver = "bicgstab"')
            solver = 'bicgstab'
        self.solver = solver

        if self.beta is not None:
            if self.beta < 0:
                raise ValueError('The inverse temperature beta has to be positive!')
        else:
            self.beta = 5

        if (mu0 < 0) or (mu0 > 1):
            raise ValueError('The sigma parameter has to be in [0,1]!')

        # values of the parameters used during the update
        self.delta_0 = delta0  # outgroup parameter
        self.mu = mu0  # sigma parameter
        self.Q = np.ones((self.L, self.N)) * mu0  # sigma parameter - posterior
        self.c = c0  # sparsity coefficient
        self.s = np.zeros(self.N, dtype = float)  # ranking scores

        # values of the parameters in the previous iteration
        self.delta_0_old = delta0  # outgroup parameter
        self.mu_old = mu0  # sigma parameter
        self.Q_old = np.ones((self.L, self.N)) * mu0  # sigma parameter - posterior
        self.c_old = c0  # sparsity coefficient
        self.s_old = np.zeros(self.N, dtype = float)  # ranking scores

        # final values after convergence --> the ones that maximize the log-likelihood
        self.delta_0_f = delta0  # outgroup parameter
        self.mu_f = mu0  # sigma parameter
        self.Q_f = np.ones((self.L, self.N)) * mu0  # sigma parameter - posterior
        self.c_f = 1.  # sparsity coefficient
        self.s_f = np.zeros(self.N, dtype = float)  # ranking scores
        self.ratio_f = None  # final ratio

    def fit(self, data, nodes, mask=None):
        """
            Model directed networks by using a probabilistic generative model that assume community and
            ranking parameters. The inference is performed via EM algorithm.

            Parameters
            ----------
            data : ndarray/sptensor
                   Graph adjacency tensor.
            nodes : list
                    List of nodes IDs.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            Iterable of dictionaries containing:
                s_f : ndarray
                      Ranking scores vector.
                u_f : ndarray
                      Out-going membership matrix.
                v_f : ndarray
                      In-coming membership matrix.
                w_f : ndarray
                      Affinity tensor.
                c_f : float
                      Sparsity coefficient.
                beta_f : float
                         Inverse temperature parameter.
                gamma_f : float
                          Ranking regularization parameter.
                mu_f : float
                       Prior sigma parameter.
                Q_f : ndarray
                      Posterior sigma parameters.
                delta0_f : float
                            Out-group interaction parameter.
                maxL : float
                       Maximum log-likelihood.
                K : int
                    Number of communities.
                nodes_s : ndarray
                          Permuted node list according to inferred scores.
                nodes_c : ndarray
                          Node list.
                seed : int
                       Realization seed.
                convergence : bool
                              Realization convergence flag.
                maxit : int
                        Realization number of iteration.
                constrained : bool
                              Realization flag for u,v,w regularization.
        """

        self.model = '_XOR'

        # initialization of the SR model
        self.SR = SR.SpringRank(N = self.N, L = self.L, solver = self.solver, gamma = self.gamma, l0 = self.l0,
                                l1 = self.l1, inf = self.inf, verbose = self.verbose, get_beta = False,
                                out_inference = False, out_folder = self.out_folder, label = self.label)

        # pre-processing of the data to handle the sparsity
        data = MT.preprocess(data, self.verbose)
        # save positions of the nonzero entries - tuple of np.ndarrays
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        for r in range(self.N_real):

            # initialization of the random state
            prng = np.random.RandomState(self.rseed)
            # initialization of the maximum log-likelihood
            maxL = -self.inf

            # Initialize all variables
            self._initialize(prng = prng)
            self._update_old_variables()
            self._update_cache(data, subs_nz, mask = mask)

            # Convergence local variables
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            if self.verbose == 2:
                print(f'\n\nUpdating realization {r} ...', end = '\n\n')
            time_start = time.time()
            loglik_values = []
            # --- single step iteration update ---
            while not convergence and it < self.max_iter:
                # main EM update: updates latent variables and calculates max difference new vs old
                _ = self._update_em(data, subs_nz, mask = mask)

                it, loglik, coincide, convergence = self._check_for_convergence(data, it, loglik, coincide, convergence,
                                                                                subs_nz, mask = mask)
                loglik_values.append(loglik)

                if self.verbose == 2:
                    print(f'Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - '
                          f'time = {np.round(time.time() - time_start, 2)} seconds')

            if self.verbose:
                print(colored('End of the realization.', 'green'),
                      f'Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - '
                      f'time = {np.round(time.time() - time_start, 2)} seconds')

            if maxL < loglik:
                maxL = loglik
                conv = convergence
                self.final_it = it
                self._update_optimal_parameters()
            self.rseed += prng.randint(100000000)

            self.maxL = maxL
            if self.final_it == self.max_iter and not conv:
                # convergence not reached
                print(colored(
                    'Solution failed to converge in {0} EM steps for realization n.{1}!'.format(self.max_iter, r),
                    'blue'))
            # end cycle over realizations

            yield {
                's': self.s_f, 'c': self.c_f, 'beta': self.beta, 'gamma': self.gamma,
                'u': self.u_f, 'v': self.v_f, 'w': self.w_f,
                'Q': self.Q_f, 'ratio': self.mu_f,
                'delta0': self.delta_0_f, 'K': self.K,
                'nodes_s': np.argsort(self.s_f)[::-1], 'nodes_c': nodes,
                'seed': self.rseed, 'logL': self.maxL, 'convergence': conv,
                'maxit': self.final_it, 'constrained': self.constrained
                }

    def _initialize(self, prng=None):
        """
            Random initialization of the latent parameters.

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)

        self._randomize_c(prng = prng)
        self._randomize_delta_0(prng = prng)

        if self.initialization == 0:
            if self.verbose > 0:
                print('Variables s, u, v, w, Q are initialized randomly.')
            self._randomize_s(prng = prng)
            self._randomize_w(prng = prng)
            self._randomize_u_v(prng = prng)
            self._randomize_means(prng = prng)

        elif self.initialization > 0:
            if self.verbose > 0:
                print('Selected initialization of s, u, v, w: from file.')
            try:
                if not self.fix_scores:
                    raise ValueError('Flag fix_scores set to False!')
                self._initialize_s(self.input_s)
                if self.verbose == 2:
                    print('s initialized from ', self.input_s)
            except:
                self._randomize_s(prng = prng)
                if self.verbose == 2:
                    print('Error: s initialized randomly.')
            try:
                if not self.fix_communities:
                    raise ValueError('Flag fix_communities set to False!')
                self._initialize_w(self.input_w)
                if self.verbose == 2:
                    print('w initialized from ', self.input_w)
            except:
                self._randomize_w(prng = prng)
                if self.verbose == 2:
                    print('Error: w initialized randomly.')
            try:
                if not self.fix_communities:
                    raise ValueError('Flag fix_communities set to False!')
                self._initialize_u_v(self.input_u, self.input_v)
                if self.verbose == 2:
                    print('u and v initialized from ', self.input_u, self.input_v)
            except:
                self._randomize_u_v(prng = prng)
                if self.verbose == 2:
                    print('Error: u, v initialized randomly.')

            if self.initialization == 2:
                if self.verbose == 2:
                    print('Selected initialization of Q: from file.')
                self._initialize_means(self.input_Q)
                if self.verbose == 2:
                    print('Q initialized from ', self.input_Q)
            else:
                if self.verbose == 2:
                    print('Error: Q initialized randomly.')
                self._randomize_means(prng = prng)

    def _randomize_c(self, prng=None, a=0.01, b=1e4):
        """
            Generate a random number in (a, b).

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)
        self.c = (b - a) * prng.random_sample(1)[0] + a

    def _randomize_means(self, prng=None, a=0.1, b=0.9):
        """
            Generate a random number in (a, b).

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if not self.fix_means:
            if prng is None:
                prng = np.random.RandomState(self.rseed)
            if self.randomize_mu:
                self.mu = (b - a) * prng.random_sample(1)[0] + a
                self.Q += self.mu - self.Q.mean()
                self.Q[self.Q > 1] = 0.99
                self.Q[self.Q < 0] = 2 * EPS
            else:
                self.Q = (b - a) * prng.random_sample(self.Q.shape) + a
                if not self.fix_mu:
                    self.mu = np.mean(self.Q)

    def _randomize_delta_0(self, prng=None, a=1e-3, b=0.5):
        """
            Generate a random number in (a, b).

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if not self.fix_delta:
            if prng is None:
                prng = np.random.RandomState(self.rseed)
            self.delta_0 = (b - a) * prng.random_sample(1)[0] + a

    def _randomize_s(self, prng=None):
        """
            Assign a random number in [-inf, +inf] to each entry of the affinity tensor s.

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)
        self.s = (1 - 2 * prng.binomial(1, .5, self.s.shape)) * prng.random_sample(self.s.shape)

    def _initialize_means(self, infile_name, prng=None):
        """
            Initialize a posteriori sigma parameters Q from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        with open(infile_name, 'rb') as f:
            dfQ = pd.read_csv(f, sep = '\s+', header = None, squeeze = True)
            self.Q = dfQ.values.T[np.newaxis, :]

        if prng is None:
            prng = np.random.RandomState(self.rseed)
        # Add noise to the initialization
        self.Q[self.Q == 1] -= self.err * 0.001 * prng.random_sample(self.Q[self.Q == 1].shape)
        self.Q[self.Q == 0] += self.err * 0.001 * prng.random_sample(self.Q[self.Q == 0].shape)
        self.mu = np.mean(self.Q)

    def _initialize_s(self, infile_name, prng=None):
        """
            Initialize ranking vector s from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        with open(infile_name, 'rb') as f:
            dfS = pd.read_csv(f, sep = '\s+', header = None)
            self.s = dfS.values
            self.s = self.s.flatten()

        # Add noise to the initialization
        max_entry = np.max(self.s)
        if prng is None:
            prng = np.random.RandomState(self.rseed)
        self.s += max_entry * self.err * 0.001 * prng.random_sample(self.s.shape)

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.s_old = np.copy(self.s)
        self.c_old = np.copy(self.c)
        self.Q_old = np.copy(self.Q)
        self.mu_old = np.copy(self.mu)
        self.delta_0_old = np.copy(self.delta_0)
        self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
        self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
        self.w_old[self.w > 0] = np.copy(self.w[self.w > 0])

    def _update_cache(self, data, subs_nz, com=True, rank=True, probs=True, mask=None):
        """
            Update the cache used in the em_update.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            com : bool
                  Flag for updating community related cache.
            rank : bool
                   Flag for updating ranking related cache.
            probs : bool
                    Flag for updating edge probabilities related cache.
            mask : ndarray
                   Mask for cv.
        """
        if probs:
            # matrix containing Qi * Qj = Yij
            self.QQt = np.einsum('ai,aj->aij', self.Q, self.Q)
            low_values_indices = self.QQt < EPS  # values are too low
            self.QQt[low_values_indices] = EPS
            # matrix containing Q_i for every j + Q_j for every i
            self.Qs = np.vstack([self.Q] * self.N) + np.hstack([self.Q.T] * self.N)
            low_values_indices = self.Qs < EPS  # values are too low
            self.Qs[low_values_indices] = EPS
            self.Qs = self.Qs[np.newaxis, :, :]
            # matrix containing QQt - (Q_i for every j + Q_j for every i) + 1 = X - Y
            self.XmY = self.QQt - self.Qs + 1

            if np.logical_or(self.QQt < 0, self.QQt > 1).any():
                print(self.QQt[np.logical_or(self.QQt < 0, self.QQt > 1)])

            if mask is not None:
                # compute masked values of X - Y for community updates
                self.XmY_masked = np.zeros_like(self.QQt)
                self.XmY_masked[mask] = self.XmY[mask]
        if rank:
            # compute s_i - s_j
            self.Ds = self._Ds()
            # compute full SR exponential term
            self.eH = self._eH()
        if com:
            # compute MT means for nonzero values
            self.M_nz = self._M_nz(subs_nz)
            # compute auxiliary variables
            self.data_hat_Mnz = self._data_hat_Mnz(data, subs_nz)

    def _Ds(self):
        """
            Compute the ranking differences. Uses an external function in order
            to speed up computations with Numba.

            Returns
            -------
            delta_s : ndarray
                      Ranking differences matrix NxN, zero for null data entries.
        """

        delta_s = delta_scores(self.N, self.s)
        return delta_s

    def _eH(self):
        """
            Compute the SR mean exponential term for all entries.

            Returns
            -------
            eH : ndarray
                 SR mean exponential term matrix NxN.
        """

        return np.exp(-0.5 * self.beta * np.power(self.Ds - self.l1, 2))

    def _data_hat_Mnz(self, data, subs_nz):
        """
            Compute auxiliary variable data_hat_Mnz = data * (1 - Q) / M.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            -------
            data_hat_Mnz : sptensor/dtensor
                           Auxiliary tensor of the same shape and type of data.
        """

        Z = np.copy(self.M_nz)
        Z[Z == 0] = 1
        if isinstance(data, skt.sptensor):
            data_hat_Mnz = data.vals * self.XmY[subs_nz] / Z
        if isinstance(data, skt.dtensor):
            data_hat_Mnz = data[subs_nz].astype('float') * self.XmY[subs_nz] / Z
        data_hat_Mnz[data_hat_Mnz == np.inf] = self.inf

        return data_hat_Mnz

    def _data_tilde(self, data, subs_nz):
        """
            Compute auxiliary variable data_tilde = data * Q.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            -------
            data_tilde : scipy/ndarray
                         Auxiliary matrix, 2-dimensional.
        """

        if self.L > 1:
            raise NotImplementedError('SpringRank for tensors not implemented! Use 2-dimensional input.')

        data_tilde = np.zeros((self.N, self.N), dtype = float)[np.newaxis, :, :]
        if isinstance(data, skt.sptensor):
            data_tilde[subs_nz] = data.vals * self.QQt[subs_nz]
        elif isinstance(data, skt.dtensor):
            data_tilde[subs_nz] = data[subs_nz] * self.QQt[subs_nz]
        try:
            # convert auxiliary tensor to scipy matrix if possible
            data_tilde = scipy.sparse.csr_matrix(data_tilde[0, :, :])
        except:
            warnings.warn('The input parameter A could not be converted to scipy.sparse.csr_matrix. '
                          'Using a dense representation (numpy).')
            data_tilde = data_tilde[0, :, :]
        return data_tilde

    def _update_em(self, data, subs_nz, mask=None):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            d_s : float
                  Maximum distance between the old and the new scores vector s.
            d_u : float
                  Maximum distance between the old and the new membership matrix u.
            d_v : float
                  Maximum distance between the old and the new membership matrix v.
            d_w : float
                  Maximum distance between the old and the new affinity tensor w.
            d_c : float
                  Distance between the old and the new SR sparsity coefficient c.
            d_mu : float
                   Distance between the old and the new prior mean of sigma.
            d_Q : float
                  Distance between the old and the new posterior mean of sigma.
        """

        if not self.fix_scores:
            d_s = self._update_s(self._data_tilde(data, subs_nz))
            self._update_cache(data, subs_nz, com = False, probs = False)
        else:
            d_s = 0

        d_c = self._update_c(self._data_tilde(data, subs_nz), mask = mask)
        self._update_cache(data, subs_nz, com = False, probs = False)

        if not self.fix_communities:
            d_u = self._update_U(subs_nz, self.data_hat_Mnz, mask = mask)
            self._update_cache(data, subs_nz, rank = False, probs = False)

            d_v = self._update_V(subs_nz, self.data_hat_Mnz, mask = mask)
            self._update_cache(data, subs_nz, rank = False, probs = False)

            if self.initialization != 1:
                if not self.assortative:
                    d_w = self._update_W(subs_nz, self.data_hat_Mnz, mask = mask)
                else:
                    d_w = self._update_W_assortative(subs_nz, self.data_hat_Mnz, mask = mask)
            else:
                d_w = 0
            self._update_cache(data, subs_nz, rank = False, probs = False)
        else:
            d_u, d_v, d_w = 0, 0, 0

        if not self.fix_delta:
            d_lam = self._update_delta_0(data, subs_nz, mask = mask)
        else:
            d_lam = 0

        d_Q = self._update_Q(data)
        if not self.fix_means:
            d_mu = self._update_mu()
        else:
            d_Q = 0
            d_mu = 0

        self._update_cache(data, subs_nz, probs = 1 - self.fix_means, rank = False, mask = mask)

        return d_s, d_u, d_v, d_w, d_c, d_lam, d_mu, d_Q

    def _update_U(self, subs_nz, data, mask=None):
        """
            Update out-going membership matrix.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            data : sptensor/dtensor
                   Graph adjacency tensor.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            dist_u : float
                     Maximum distance between the old and the new membership matrix u.
        """

        self.u *= self._update_membership(data, subs_nz, self.u, self.v, self.w, 1)

        if mask is not None:
            Du = np.einsum('aij,jq->iq', self.XmY_masked, self.v)
        else:
            Du = np.einsum('aij,jq->iq', self.XmY, self.v)
        if not self.assortative:
            w_k = np.einsum('akq->kq', self.w)
            Z_uk = np.einsum('iq,kq->ik', Du, w_k)
        else:
            w_k = np.einsum('ak->k', self.w)
            Z_uk = np.einsum('ik,k->ik', Du, w_k)

        if not self.constrained:
            non_zeros = Z_uk > EPS
            self.u[Z_uk < EPS] = 0.
            self.u[non_zeros] /= Z_uk[non_zeros]
        else:
            self.u /= Z_uk + self.delta_u

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.
        assert (self.u <= self.inf).all()

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V(self, subs_nz, data, mask=None):
        """
            Update in-coming membership matrix.
            Same as _update_U but with:
            data <-> data_T
            w <-> w_T
            u <-> v

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            data : sptensor/dtensor
                   Graph adjacency tensor.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            dist_v : float
                     Maximum distance between the old and the new membership matrix v.
        """

        self.v *= self._update_membership(data, subs_nz, self.u, self.v, self.w, 2)

        if mask is not None:
            Dv = np.einsum('aij,ik->jk', self.XmY_masked, self.u)
        else:
            Dv = np.einsum('aij,ik->jk', self.XmY, self.u)
        if not self.assortative:
            w_k = np.einsum('akq->kq', self.w)
            Z_vk = np.einsum('jk,kq->jq', Dv, w_k)
        else:
            w_k = np.einsum('ak->k', self.w)
            Z_vk = np.einsum('jk,k->jk', Dv, w_k)

        if not self.constrained:
            non_zeros = Z_vk > EPS
            self.v[Z_vk < EPS] = 0.
            self.v[non_zeros] /= Z_vk[non_zeros]
        else:
            self.v /= Z_vk + self.delta_v

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.
        assert (self.v <= self.inf).all()

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_W(self, subs_nz, data, mask=None):
        """
            Update affinity tensor.

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            data : sptensor/dtensor
                   Graph adjacency tensor.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """

        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])

        uttkrp_I = data[:, np.newaxis, np.newaxis] * UV
        for _, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights = uttkrp_I[:, k, q], minlength = self.L)

        self.w *= uttkrp_DKQ

        if mask is not None:
            Z = np.einsum('aij,ik,jq->akq', self.XmY_masked, self.u, self.v)
        else:
            Z = np.einsum('aij,ik,jq->akq', self.XmY, self.u, self.v)
        if not self.constrained:
            non_zeros = Z > 0
            self.w[non_zeros] /= Z[non_zeros]
        else:
            self.w /= Z + self.delta_w

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        assert (self.w <= self.inf).all()

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative(self, subs_nz, data, mask=None):
        """
            Update affinity tensor (assuming assortativity).

            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            data : sptensor/dtensor
                   Graph adjacency tensor.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])

        uttkrp_I = data[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights = uttkrp_I[:, k], minlength = self.L)

        self.w *= uttkrp_DKQ

        if mask is not None:
            Z = np.einsum('aij,ik,jk->ak', self.XmY_masked, self.u, self.v)
        else:
            Z = np.einsum('aij,ik,jk->ak', self.XmY, self.u, self.v)
        if not self.constrained:
            non_zeros = Z > 0
            self.w[non_zeros] /= Z[non_zeros]
        else:
            self.w /= Z + self.delta_w

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.
        assert (self.w <= self.inf).all()

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_s(self, data):
        """
            Main routine to calculate SpringRank by a solving linear system.
            If gamma != 0, performs L2 regularization.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            Returns
            -------
            dist_s : float
                    Maximum distance between the old and the new ranking vector s.
        """

        # compute ranks update
        self.s, _, _ = self.SR.fit(data)
        # compute update improvement
        dist_s = np.amax(abs(self.s - self.s_old))
        # update variables
        if isinstance(data, scipy.sparse.csr_matrix):
            self.s_old = self.s.copy()
        elif isinstance(data, np.ndarray):
            self.s_old = np.copy(self.s)

        return dist_s

    def _update_c(self, data, mask=None):
        """
            Compute the sparsity coefficient.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            dist_c : float
                Sparsity coefficient.
        """

        if mask is None:
            denominator = (self.eH * self.QQt[0]).sum()
        else:
            denominator = (self.eH * self.QQt[0])[mask[0]].sum()
        if denominator == 0:
            self.c = self.inf
        else:
            self.c = data.sum() / denominator

        # compute update improvement
        dist_c = abs(self.c - self.c_old)
        # update variable
        self.c_old = np.copy(self.c)

        return dist_c

    def _update_mu(self):
        """
            Compute the prior mean for sigma.

            Returns
            -------
            dist_mu : float
        """

        self.mu = np.mean(self.Q)

        # compute update improvement
        dist_mu = abs(self.mu - self.mu_old)
        if self.mu < self.err_max:
            self.mu = self.err_max
        if 1 - self.mu < self.err_max:
            self.mu = 1 - self.err_max
        # update variable
        self.mu_old = np.copy(self.mu)

        return dist_mu

    def _update_delta_0(self, data, subs_nz, mask=None):

        den = 2 * self.QQt - self.Qs  # X - 1 expectation
        den[-den < self.err_max] = -self.err_max

        if isinstance(data, skt.sptensor):
            self.delta_0 = (data.vals * den[subs_nz]).sum()
        elif isinstance(data, skt.dtensor):
            self.delta_0 = (data[subs_nz] * den[subs_nz]).sum()

        if mask is None:
            self.delta_0 /= den.sum()
        else:
            self.delta_0 /= den[mask].sum()

        assert (self.delta_0 <= self.inf) and (self.delta_0 > 0)
        # compute update improvement
        dist_lam = np.abs(self.delta_0 - self.delta_0_old)
        # update variable
        self.delta_0_old = np.copy(self.delta_0)

        return dist_lam

    def _update_Q(self, data):
        """
            Compute the posterior mean for sigma.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            Returns
            -------
            dist_Q : float
        """

        self.S = (self.c * self.eH)[np.newaxis, :, :]

        if self.w.ndim == 2:
            M = np.einsum('ik,jk->ijk', self.u, self.v)
            M = np.einsum('ijk,ak->aij', M, self.w)
        else:
            M = np.einsum('ik,jq->ijkq', self.u, self.v)
            M = np.einsum('ijkq,akq->aij', M, self.w)
        self.M = M

        if not self.fix_means:
            veclam = np.ones((self.L, self.N, self.N)) * self.delta_0

            if isinstance(data, skt.sptensor):
                AS = poisson.pmf(data.toarray(), self.S)
                AM = poisson.pmf(data.toarray(), self.M)
                AL = poisson.pmf(data.toarray(), veclam)
            if isinstance(data, skt.dtensor):
                AS = poisson.pmf(data, self.S)
                AM = poisson.pmf(data, self.M)
                AL = poisson.pmf(data, veclam)

            # Init
            ASt = np.einsum('ij,ji->ij', AS[0], AS[0])
            ALt = np.einsum('ij,ji->ij', AL[0], AL[0])
            AMt = np.einsum('ij,ji->ij', AM[0], AM[0])
            Qs_old = np.vstack([np.copy(self.Q_old)] * self.N)
            # j = i not influential on the final product
            np.fill_diagonal(ASt, 1.)
            np.fill_diagonal(ALt, 1.)
            np.fill_diagonal(AMt, 1.)

            L1 = Qs_old * np.log(ASt + EPS) + (1. - Qs_old) * np.log(ALt + EPS)
            L2 = (1. - Qs_old) * np.log(AMt + EPS) + Qs_old * np.log(ALt + EPS)
            L1 = L1.sum(axis = 1) + np.log(self.mu + EPS)
            L2 = L2.sum(axis = 1) + np.log(1. - self.mu + EPS)

            max_L = max(max(L1), max(L2))

            L1 -= max_L
            L2 -= max_L

            phi1 = np.exp(L1)
            phi2 = np.exp(L2)

            max_phi = max(max(phi1), max(phi2))

            phi1 /= max_phi
            phi2 /= max_phi

            self.Q[0] = phi1 / (phi1 + phi2)

            nans = np.isnan(self.Q[0])
            mask1 = np.logical_and(np.isnan(phi1), np.logical_not(np.isnan(phi2)))
            mask2 = np.logical_and(np.isnan(phi2), np.logical_not(np.isnan(phi1)))
            mask3 = np.logical_and(np.isnan(phi2), np.isnan(phi1))

            self.Q[0][nans] = 0.5
            self.Q[0][mask1] = np.finfo(np.float64).tiny
            self.Q[0][mask2] = 1 - np.finfo(np.float64).tiny
            self.Q[0][mask3] = 0.5

            if self.verbose == 2:
                print('\n\tQ update info:',
                      f'phi1 avg: {np.mean(phi1):.2e}',
                      f'phi1 max: {np.max(phi1):.2e}',
                      f'phi1 min: {np.min(phi1):.2e}',
                      f'phi2 avg: {np.mean(phi2):.2e}',
                      f'phi2 max: {np.max(phi2):.2e}',
                      f'phi2 min: {np.min(phi2):.2e}',
                      sep = '\n\t\t', end = '\n\n')

            low_values_indices = self.Q < EPS # values are too low
            self.Q[low_values_indices] = EPS
            assert (self.Q <= self.inf).all()
            if (self.Q < 0).any():
                print(self.Q[self.Q < 0])

        # compute update improvement
        dist_Q = np.max(np.abs(self.Q - self.Q_old))
        # update variable
        self.Q_old = np.copy(self.Q)

        return dist_Q

    def _check_for_convergence(self, data, it, loglik, coincide, convergence, subs_nz, t=1, mask=None):
        """
            Check for convergence by using the log-likelihood values.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            mask : ndarray
                   Mask for cv.

            Returns
            -------
            it : int
                 Number of iteration.
            loglik : float
                     Log-likelihood value.
            coincide : int
                       Number of time the update of the log-likelihood respects the tolerance.
            convergence : bool
                          Flag for convergence.
        """

        if it % t == 0:
            old_L = loglik
            loglik = self.__Likelihood(data, subs_nz, mask = mask)

            if abs(loglik - old_L) < self.tolerance:
                coincide += 1
            else:
                coincide = 0
        if coincide > self.decision:
            convergence = True
        it += 1

        return it, loglik, coincide, convergence

    def __Likelihood(self, data, subs_nz, mask=None):
        """
            Compute the log-likelihood of the data.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            mask : ndarray
                   Mask for cv.
            Returns
            -------
            l : float
                Likelihood value.
        """

        S_nz = np.copy(self.S)[subs_nz]
        S_nz[S_nz == 0] = 1
        M_nz = np.copy(self.M_nz)
        M_nz[M_nz == 0] = 1

        # compute entropy term
        H = lambda p: entropy([p, 1 - p])
        l = np.array(list(map(H, self.Q))).sum()
        if mask is not None:
            # compute mu term
            l += np.log(self.mu) * self.Q.sum() + np.log(1 - self.mu) * (1 - self.Q).sum()
            # compute linear term
            l -= (self.QQt * self.S + self.XmY * self.M + (2 * self.QQt - self.Qs) * self.delta_0)[mask].sum()
        else:
            # compute mu term
            l += np.log(self.mu) * self.Q.sum() + np.log(1 - self.mu) * (1 - self.Q).sum()
            # compute linear term
            l -= (self.QQt * self.S + self.XmY * self.M + (2 * self.QQt - self.Qs) * self.delta_0).sum()

        # compute logarithmic term on non zero elements
        if isinstance(data, skt.dtensor):
            spl = data[subs_nz] * (self.QQt[subs_nz] * np.log(S_nz) + self.XmY[subs_nz] * np.log(M_nz) + (
                    2 * self.QQt[subs_nz] - self.Qs[subs_nz]) * np.log(self.delta_0))
        if isinstance(data, skt.sptensor):
            spl = data.vals * (self.QQt[subs_nz] * np.log(S_nz) + self.XmY[subs_nz] * np.log(M_nz) + (
                    2 * self.QQt[subs_nz] - self.Qs[subs_nz]) * np.log(self.delta_0))
        l += spl.sum()
        # compute prior term on u, v
        if self.constrained:
            l -= self.lambda_u * self.u.sum() + self.lambda_v * self.v.sum() + self.lambda_w * self.w.sum()

        if not np.isnan(l):
            return l.item()
        else:
            print(colored("Likelihood is NaN!!", 'red'))
            sys.exit(1)

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """
        self.s_f = np.copy(self.s)
        self.c_f = np.copy(self.c)
        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)
        self.Q_f = np.copy(self.Q)
        self.mu_f = np.copy(self.mu)
        self.delta_0_f = np.copy(self.delta_0)

    @staticmethod
    def output_parameters(i, out, **conf):
        """
            Output results for each realization in compressed files.

            Parameters
            ----------
            i : int
                Realization ID.
            out : dict
                  Dictionary with realization output.
            conf : dict
                   Dictionary with configuration parameters.
        """

        keys = list(out.keys())[:-5]
        vals = list(out.values())[:-5]
        output_path = conf['out_folder'] + 'parameters_' + conf['label'] + '_XOR_' + str(out['seed'])
        # save to compressed file latent variables and number of communities
        np.savez_compressed(output_path + '.npz', **dict(zip(keys, vals)))

        if conf['verbose']:
            print()
            print(f'It #{i}: Parameters saved in: {output_path}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]', end = '\n\n')

    @staticmethod
    def output_csv(out, mask=None, **conf):
        """
            Output experiment statistics for each realization in csv file.

            Parameters
            ----------
            out : dict
                  Dictionary with realization output.
            conf : dict
                   Dictionary with configuration parameters.
            mask : ndarray
                   Mask for cv.

        """
        metrics_path = conf['out_folder'] + 'metrics_' + conf['label'] + '_XOR'
        save_metrics(out, conf['in_folder'], metrics_path, model = 'XOR', mask = np.logical_not(mask),
                     clas = conf['classification'], cv = conf['cv'], ground_truth = conf['gt'])

        if conf['verbose']:
            print(f'Metrics saved in: {metrics_path}.csv')
            print('Load as a pandas dataframe.', end = '\n\n')

    @classmethod
    def save_results(cls, out, mask=None, **conf):
        """

            Parameters
            ----------
            out : dict
                  Dictionary with realization output.
            conf : dict
                   Dictionary with configuration parameters.
            mask : ndarray
                   Mask for cv.
        """

        for i, d in enumerate(out):
            cls.output_parameters(i, d, **conf)
            cls.output_csv(d, mask = mask, **conf)
