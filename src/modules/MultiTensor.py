"""
    Class definition of MultiTensor.

    Sparse input data are handled through the scikit-tensor library, Version 0.1
    ("Maximilian Nickel. Available Online, November 2013."), which represents
    sparse matrices via the Coordinate Format (COO) [subs, vals].

    Implementation of the coordinate ascent algorithm corresponding Expectation
    Maximization on the MultiTensor log-likelihood.
"""

from __future__ import print_function

import time
import numpy as np
import pandas as pd
import sktensor as skt

from termcolor import colored
from compute_metrics import save_metrics

class MultiTensor(object):

    def __init__(self, N=100, L=1, K=2, initialization=0, rseed=42, inf=1e10, err_max=1e-8, err=0.01, N_real=1,
                 tolerance=0.001, decision=10, max_iter=500, out_inference=False, out_folder='../data/output/',
                 in_folder=None, label='', assortative=False, verbose=0, input_u='../data/input/u.dat',
                 input_v='../data/input/v.dat', input_w='../data/input/w.dat', constrained=False,
                 lambda_u=5., lambda_v=5., lambda_w=10., cv=False, gt=False, **kargs):

        self.N = N  # number of nodes
        self.L = L # number of layers
        self.K = K  # number of communities
        self.rseed = rseed  # random seed for the initialization
        self.inf = inf  # initial value of the log-likelihood
        self.err_max = err_max  # minimum value for the parameters
        self.err = err  # noise for the initialization
        self.N_real = N_real  # number of iterations with different random initialization
        self.tolerance = tolerance  # tolerance parameter for convergence
        self.decision = decision  # convergence parameter
        self.max_iter = max_iter  # maximum number of EM steps before aborting
        self.out_inference = out_inference  # flag for storing the inferred parameters
        self.out_folder = out_folder  # path for storing the output
        self.in_folder = in_folder  # path for reading the labels
        self.label = label # additional label for the output
        self.assortative = assortative  # if True, the network is assortative
        self.input_u = input_u  # path of the input file u (when initialization=1)
        self.input_v = input_v  # path of the input file v (when initialization=1)
        self.input_w = input_w  # path of the input file w (when initialization=1)
        self.constrained = constrained # flag for performing constrained optimization on u, v with lagrange multipliers
        self.lambda_u = lambda_u # mean for the exponential prior on u
        self.lambda_v = lambda_v # mean for the exponential prior on v
        self.lambda_w = lambda_w # mean for the exponential prior on w
        self.cv = cv # flag for including cv metrics in the output
        self.gt = gt # flat for including metrics wrt ground truth in the output
        if verbose > 2 or not isinstance(verbose, int): # verbosity indicator
            raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
        self.verbose = verbose
        if initialization not in {0, 1,2}:  # indicator for choosing how to initialize s, u, v and w
            raise ValueError('The initialization parameter can be either 0 or 1. It is used as an indicator to '
                             'initialize the ranking vector s, the membership matrices u and v and the affinity matrix w.'
                             'If it is 0, they will be generated randomly, otherwise they will upload from file.')
        self.initialization = initialization

        if self.initialization == 1:
            try:
                dfU = pd.read_csv(self.input_u, sep='\s+', header=None)
                self.N, self.K = dfU.shape
                #self.N_real = 1
            except:
                if verbose==2: print('Input file for u non found. Using dimensions passed as input.')

        # values of the parameters used during the update
        self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.K), dtype=float)  # in-coming membership

        # values of the parameters in the previous iteration
        self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-coming membership

        # final values after convergence --> the ones that maximize the log-likelihood
        self.u_f = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v_f = np.zeros((self.N, self.K), dtype=float)  # in-coming membership

        # values of the affinity tensor
        if self.assortative:  # purely diagonal matrix
            self.w = np.zeros((self.L, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K), dtype=float)
        else:
            self.w = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
            self.w_f = np.zeros((self.L, self.K, self.K), dtype=float)

    def fit(self, data, nodes, mask=None):
        """
            Model directed networks by using a probabilistic generative model that assume community
            parameters. The inference is performed via EM algorithm.

            Parameters
            ----------
            data : ndarray/sptensor
                   Graph adjacency tensor.
            nodes : list
                    List of nodes IDs.
            Returns
            ----------
            u_f : ndarray
                  Out-going membership matrix.
            v_f : ndarray
                  In-coming membership matrix.
            w_f : ndarray
                  Affinity tensor.
            maxL : float
                   Maximum log-likelihood.
        """

        # initialization of the random state
        prng = np.random.RandomState(self.rseed)

        # pre-processing of the data to handle the sparsity
        data = preprocess(data, self.verbose)
        # save positions of the nonzero entries
        if isinstance(data, skt.dtensor):
            subs_nz = data.nonzero()
        elif isinstance(data, skt.sptensor):
            subs_nz = data.subs

        for r in range(self.N_real):
            # initialization of the maximum log-likelihood
            maxL = -self.inf
            self.final_it = None


            # Initialize all variables
            self._initialize(prng=np.random.RandomState(self.rseed))
            self._update_old_variables()
            self._update_cache(data, subs_nz)

            # Convergence local variables
            coincide, it = 0, 0
            convergence = False
            loglik = self.inf

            if self.verbose == 2:
                print(f'\n\nUpdating realization {r} ...', end='\n\n')
            time_start = time.time()
            loglik_values = []
            # --- single step iteration update ---
            while not convergence and it < self.max_iter:
                # main EM update: updates memberships and calculates max difference new vs old
                delta_u, delta_v, delta_w = self._update_em(data, subs_nz)

                it, loglik, coincide, convergence = self._check_for_convergence(data, it, loglik, coincide, convergence, subs_nz, mask=mask)
                loglik_values.append(loglik)

                if self.verbose == 2:
                    print('done!')
                    print(f'Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - '
                          f'time = {np.round(time.time() - time_start, 2)} seconds')

            if self.verbose:
                print(colored('End of the realization.', 'green'), f'Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - '
                f'time = {np.round(time.time() - time_start, 2)} seconds')

            if maxL < loglik:
                self._update_optimal_parameters()
                maxL = loglik
                self.final_it = it
                conv = convergence
            self.rseed += prng.randint(100000000)

            self.maxL = maxL
            if self.final_it == self.max_iter and not conv:
                # convergence not reached
                print(colored('Solution failed to converge in {0} EM steps for realization n.{1}!'.format(self.max_iter, r), 'blue'))
            # end cycle over realizations

            yield {
                'u': self.u_f, 'v': self.v_f, 'w': self.w_f,
                'K': self.K, 'nodes_c': nodes,
                'seed': self.rseed, 'logL': self.maxL,
                'convergence': conv, 'maxit': self.final_it,
                'constrained': self.constrained
                }


    def _initialize(self, prng=None):
        """
            Random initialization of the parameters u, v, w.

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)

        if self.initialization == 0:
            if self.verbose == 1:
                print('Latent variables u, v, w are initialized randomly.')
            self._randomize_w(prng=prng)
            self._randomize_u_v(prng=prng)

        elif self.initialization == 1:
            if self.verbose == 1:
                print('Selected initialization of u, v, w: from file.')
            try:
                self._initialize_w(self.input_w)
                if self.verbose == 2:
                    print('w initialized from ',self.input_w)
            except:
                self._randomize_w(prng=prng)
                if self.verbose == 2:
                    print('Input file not found: w initialized randomly.')
            try:
                self._initialize_u_v(self.input_u, self.input_v)
                if self.verbose == 2:
                    print('u and v initialized from ',self.input_u,self.input_v)
            except:
                self._randomize_u_v(prng=prng)
                if self.verbose == 2:
                    print('Input files not found: u, v initialized randomly.')

    def _randomize_w(self, prng = None):
        """
            Assign a random number in (0, 1.) to each entry of the affinity tensor w.

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)
        for i in range(self.L):
            for k in range(self.K):
                if self.assortative:
                    self.w[i, k] = prng.random_sample(1)
                else:
                    for q in range(k, self.K):
                        if q == k:
                            self.w[i, k, q] = prng.random_sample(1)
                        else:
                            self.w[i, k, q] = self.w[i, q, k] = self.err * prng.random_sample(1)

    def _randomize_u_v(self, prng = None):
        """
            Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.

            Parameters
            ----------
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)

        self.u = prng.random_sample(self.u.shape)
        row_sums = self.u.sum(axis=1)
        self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

        self.v = prng.random_sample(self.v.shape)
        row_sums = self.v.sum(axis=1)
        self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _initialize_u_v(self, infile_u, infile_v, prng = None):
        """
            Initialize out-going and in-coming membership matrix (u or v) from file.

            Parameters
            ----------
            infile_u : str
                       Path of the u input file.
            infile_v : str
                       Path of the v input file.
           prng : RandomState
                 Container for the Mersenne Twister pseudo-random number generator.
        """

        if prng is None:
            prng = np.random.RandomState(self.rseed)

        with open(infile_u, 'rb') as f:
            dfU = pd.read_csv(f, sep='\s+', header=None)
            self.u = dfU.values

        max_entry = np.max(self.u)
        # Add noise to the initialization
        self.u += max_entry * self.err * prng.random_sample(self.u.shape)

        with open(infile_v, 'rb') as f:
            dfV = pd.read_csv(f, sep='\s+', header=None)
            self.v = dfV.values

        max_entry = np.max(self.v)
        # Add noise to the initialization
        self.v += max_entry * self.err * prng.random_sample(self.v.shape)

    def _initialize_w(self, infile_name, prng = None):
        """
            Initialize affinity tensor w from file.

            Parameters
            ----------
            infile_name : str
                          Path of the input file.
            prng : RandomState
                  Container for the Mersenne Twister pseudo-random number generator.
        """

        with open(infile_name, 'rb') as f:
            dfW = pd.read_csv(f, sep='\s+', header=None)
            if self.assortative:
                self.w = np.diag(dfW).copy()
                if self.L == 1:
                    self.w = self.w[np.newaxis, :]
            else:
                self.w = dfW.values
                if self.L == 1:
                    self.w = self.w[np.newaxis, :, :]


        max_entry = np.max(self.w)
        # Add noise to the initialization
        if prng is None:
            prng = np.random.RandomState(self.rseed)
        self.w += max_entry * self.err * prng.random_sample(self.w.shape)

    def _update_old_variables(self):
        """
            Update values of the parameters in the previous iteration.
        """

        self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
        self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
        self.w_old[self.w > 0] = np.copy(self.w[self.w > 0])

    def _update_cache(self, data, subs_nz):
        """
            Update the cache used in the em_update.
            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            data_T_vals : ndarray
                          Array with values of entries A[j, i] given non-zero entry (i, j).
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
        """

        self.M_nz = self._M_nz(subs_nz)
        self.M_nz[self.M_nz == 0] = 1
        if isinstance(data, skt.dtensor):
            self.data_M_nz = data[subs_nz] / self.M_nz
        elif isinstance(data, skt.sptensor):
            self.data_M_nz = data.vals / self.M_nz

    def _M_nz(self, subs_nz):
        """
            Compute the mean lambda0_ij for only non-zero entries.
            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            -------
            nz_recon_I : ndarray
                         Mean lambda0_ij for only non-zero entries.
        """

        if not self.assortative:
            nz_recon_IQ = np.einsum('Ik,Ikq->Iq', self.u[subs_nz[1], :], self.w[subs_nz[0], :, :])
        else:
            nz_recon_IQ = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.w[subs_nz[0], :])
        nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, self.v[subs_nz[2], :])

        return nz_recon_I

    def _update_em(self, data, subs_nz):
        """
            Update parameters via EM procedure.

            Parameters
            ----------
            data : sptensor/dtensor
                   Graph adjacency tensor.
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            ----------
            d_u : float
                  Maximum distance between the old and the new membership matrix u.
            d_v : float
                  Maximum distance between the old and the new membership matrix v.
            d_w : float
                  Maximum distance between the old and the new affinity tensor w.
        """

        d_u = self._update_U(subs_nz, self.data_M_nz)
        self._update_cache(data, subs_nz)

        d_v = self._update_V(subs_nz, self.data_M_nz)
        self._update_cache(data, subs_nz)

        if self.initialization != 1:
            if not self.assortative:
                d_w = self._update_W(subs_nz, self.data_M_nz)
            else:
                d_w = self._update_W_assortative(subs_nz, self.data_M_nz)
        else:
            d_w = 0
        self._update_cache(data, subs_nz)

        return d_u, d_v, d_w

    def _update_U(self, subs_nz, data):
        """
            Update out-going membership matrix.
            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            -------
            dist_u : float
                     Maximum distance between the old and the new membership matrix u.
        """

        self.u *= self._update_membership(data, subs_nz, self.u, self.v, self.w, 1)

        Du = np.einsum('iq->q', self.v)
        if not self.assortative:
            w_k = np.einsum('akq->kq', self.w)
            Z_uk = np.einsum('q,kq->k', Du, w_k)
        else:
            w_k = np.einsum('ak->k', self.w)
            Z_uk = np.einsum('k,k->k', Du, w_k)

        if not self.constrained:
            non_zeros = Z_uk > 0.
            self.u[:, Z_uk == 0] = 0.
            self.u[:, non_zeros] /= Z_uk[np.newaxis, non_zeros]
        else:
            self.u /= Z_uk[np.newaxis, :] + self.lambda_u

        low_values_indices = self.u < self.err_max  # values are too low
        self.u[low_values_indices] = 0.  # and set to 0.

        dist_u = np.amax(abs(self.u - self.u_old))
        self.u_old = np.copy(self.u)

        return dist_u

    def _update_V(self, subs_nz, data):
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
            Returns
            -------
            dist_v : float
                     Maximum distance between the old and the new membership matrix v.
        """

        self.v *= self._update_membership(data, subs_nz, self.u, self.v, self.w, 2)

        Dv = np.einsum('iq->q', self.u)
        if not self.assortative:
            w_k = np.einsum('aqk->qk', self.w)
            Z_vk = np.einsum('q,qk->k', Dv, w_k)
        else:
            w_k = np.einsum('ak->k', self.w)
            Z_vk = np.einsum('k,k->k', Dv, w_k)

        if not self.constrained:
            non_zeros = Z_vk > 0
            self.v[:, Z_vk == 0] = 0.
            self.v[:, non_zeros] /= Z_vk[np.newaxis, non_zeros]
        else:
            self.v /= Z_vk[np.newaxis, :] + self.lambda_v

        low_values_indices = self.v < self.err_max  # values are too low
        self.v[low_values_indices] = 0.  # and set to 0.

        dist_v = np.amax(abs(self.v - self.v_old))
        self.v_old = np.copy(self.v)

        return dist_v

    def _update_W(self, subs_nz, data):
        """
            Update affinity tensor.
            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """

        sub_w_nz = self.w.nonzero()
        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = data[:, np.newaxis, np.newaxis] * UV
        for a, k, q in zip(*sub_w_nz):
            uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k, q], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = np.einsum('k,q->kq', self.u.sum(axis=0), self.v.sum(axis=0))[np.newaxis, :, :]
        if not self.constrained:
            non_zeros = Z > 0
            self.w[non_zeros] /= Z[non_zeros]
        else:
            self.w /= Z + self.lambda_w

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_W_assortative(self, subs_nz, data):
        """
            Update affinity tensor (assuming assortativity).
            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            Returns
            -------
            dist_w : float
                     Maximum distance between the old and the new affinity tensor w.
        """

        uttkrp_DKQ = np.zeros_like(self.w)

        UV = np.einsum('Ik,Ik->Ik', self.u[subs_nz[1], :], self.v[subs_nz[2], :])
        uttkrp_I = data[:, np.newaxis] * UV
        for k in range(self.K):
            uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L)

        self.w *= uttkrp_DKQ

        Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))[np.newaxis, :]
        if not self.constrained:
            non_zeros = Z > 0
            self.w[non_zeros] /= Z[non_zeros]
        else:
            self.w /= Z + self.lambda_w

        low_values_indices = self.w < self.err_max  # values are too low
        self.w[low_values_indices] = 0.  # and set to 0.

        dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

        return dist_w

    def _update_membership(self, data, subs_nz, u, v, w, m):
        """
            Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.
            Parameters
            ----------
            subs_nz : tuple
                      Indices of elements of data that are non-zero.
            u : ndarray
                Out-going membership matrix.
            v : ndarray
                In-coming membership matrix.
            w : ndarray
                Affinity tensor.
            m : int
                Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
                works with the matrix u; if 2 it works with v.
            Returns
            -------
            uttkrp_DK : ndarray
                        Matrix which is the result of the matrix product of the unfolding of the tensor and the
                        Khatri-Rao product of the membership matrix.
        """

        if not self.assortative:
            uttkrp_DK = sp_uttkrp(data, subs_nz, m, u, v, w)
        else:
            uttkrp_DK = sp_uttkrp_assortative(data, subs_nz, m, u, v, w)

        return uttkrp_DK

    def _check_for_convergence(self, data, it, loglik, coincide, convergence, subs_nz, mask=None):
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

        if it % 10 == 0:
            old_L = loglik
            loglik = self.__Likelihood(data, subs_nz, mask=mask)

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
            Returns
            -------
            l : float
                Pseudo log-likelihood value.
        """

        M = self._M_full(self.u, self.v, self.w)
        if mask is not None:
            M = M[mask]
        logM = np.log(self.M_nz)
        if isinstance(data, skt.dtensor):
            l = -M.sum() + (data[subs_nz] * logM).sum()
        elif isinstance(data, skt.sptensor):
            l = -M.sum() + (data.vals * logM).sum()
        if self.constrained:
            l-= self.lambda_u * self.u.sum() + self.lambda_v * self.v.sum() + self.lambda_w * self.w.sum()

        if np.isnan(l):
            print(colored("Likelihood is NaN!!", 'red'))
            sys.exit(1)
        else:
            return l.item()

    def _M_full(self, u, v, w):
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
             Mean M for all entries.
        """

        if w.ndim == 2:
            # assortative case
            M = np.einsum('ik,jk->ijk', u, v)
            M = np.einsum('ijk,ak->aij', M, w)
        else:
            M = np.einsum('ik,jq->ijkq', u, v)
            M = np.einsum('ijkq,akq->aij', M, w)
        return M

    def _update_optimal_parameters(self):
        """
            Update values of the parameters after convergence.
        """

        self.u_f = np.copy(self.u)
        self.v_f = np.copy(self.v)
        self.w_f = np.copy(self.w)

    @staticmethod
    def output_parameters(i, out, **conf):
        """
            Output results for each realization in compressed files.

            Parameters
            ----------
            i
            out
            conf
        """

        keys = list(out.keys())[:-5]
        vals = list(out.values())[:-5]
        output_path = conf['out_folder'] + 'parameters_' + conf['label'] + '_MT_' + str(out['seed'])
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
            out
            mask
            conf
        """

        metrics_path = conf['out_folder'] + 'metrics_' + conf['label'] + '_MT'
        save_metrics(out, conf['in_folder'], metrics_path, model = 'MT', mask = np.logical_not(mask),
                     clas = conf['classification'], cv = conf['cv'], ground_truth = conf['gt'])

        if conf['verbose']:
            print(f'Metrics saved in: {metrics_path}.csv')
            print('Load as a pandas dataframe.', end = '\n\n')

    @classmethod
    def save_results(cls, out, mask=None, **conf):
        """

            Parameters
            ----------
            out
            mask
            conf
        """

        for i, d in enumerate(out):
            cls.output_parameters(i, d, **conf)
            cls.output_csv(d, mask = mask, **conf)



def sp_uttkrp(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version).
        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= (w[subs[0], k, :].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
        elif m == 2:  # we are updating v
            tmp *= (w[subs[0], :, k].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
    """
        Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.
        Parameters
        ----------
        vals : ndarray
               Values of the non-zero entries.
        subs : tuple
               Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
               equal to the dimension of tensor.
        m : int
            Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
            works with the matrix u; if 2 it works with v.
        u : ndarray
            Out-going membership matrix.
        v : ndarray
            In-coming membership matrix.
        w : ndarray
            Affinity tensor.
        Returns
        -------
        out : ndarray
              Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
              of the membership matrix.
    """

    if m == 1:
        D, K = u.shape
        out = np.zeros_like(u)
    elif m == 2:
        D, K = v.shape
        out = np.zeros_like(v)

    for k in range(K):
        tmp = vals.copy()
        if m == 1:  # we are updating u
            tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)
        elif m == 2:  # we are updating v
            tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

    return out


def get_item_array_from_subs(A, ref_subs):
    """
        Get values of ref_subs entries of a dense tensor.
        Output is a 1-d array with dimension = number of non zero entries.
    """

    return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


def preprocess(X, verbose):
    """
        Pre-process input data tensor.
        If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

        Parameters
        ----------
        X : ndarray
            Input data (tensor).
        verbose : int
                  Verbosity level.
        Returns
        ----------
        X : sptensor/dtensor
            Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if not X.dtype == np.dtype(int).type:
        X = X.astype(int)
    if isinstance(X, np.ndarray) and is_sparse(X):
        X = sptensor_from_dense_array(X)
        if verbose == 2:
            print('Using sparse representation for input data.')
    else:
        X = skt.dtensor(X)
        if verbose == 2:
            print('Using dense representation for input data.')

    return X


def is_sparse(X):
    """
        Check whether the input tensor is sparse.
        It implements a heuristic definition of sparsity. A tensor is considered sparse if:
        given
        M = number of modes
        S = number of entries
        I = number of non-zero entries
        then
        N > M(I + 1)
        Parameters
        ----------
        X : ndarray
            Input data.
        Returns
        -------
        Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
        Create an sptensor from a ndarray or dtensor.
        Parameters
        ----------
        X : ndarray
            Input data.
        Returns
        -------
        sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs]

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)
