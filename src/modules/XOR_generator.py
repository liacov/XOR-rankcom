"""
    Class for generation and management of synthetic single-layer networks according to the XOR model.
    It assumes a mixed effect of the community and hierarchical latent structures.

    Possible options: model with s permuted, model with s not permuted.
"""

import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse

from numba import jit


class SyntNetXOR(object):

    def __init__(self, m=1, N=100, K=3, l=1, prng=42, avg_degree=10., mu=0.5, structure='assortative', label='test',
                 beta=1e4, gamma=0.5, delta0=0.01, eta=0.5, ag=0.6, bg=1., corr=0., over=0., means=(), stds=(),
                 verbose=0, folder='../../data/input', L1=False, output_parameters=False, output_adj=False,
                 outfile_adj='None', use_leagues=False, permute=True):
        self.N = N # network size (node number)
        self.m = m # number of networks to be generated
        self.prng = prng # seed random number generator
        self.label = label # label (associated uniquely with the set of inputs)
        self.folder = folder # input data folder path
        self.output_parameters = output_parameters # flag for storing the parameters
        self.output_adj = output_adj # flag for storing the generated adjacency matrix
        self.outfile_adj = outfile_adj # name for saving the adjacency matrix
        self.avg_degree = avg_degree # required average degree
        self.delta0 = delta0 # outgroup interaction probability
        self.permute = permute # flag for permuting s variables (not overlapping option)

        if verbose > 2 and not isinstance(verbose, int):
            raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
        self.verbose = verbose # verbosity flag

        if mu < 0 or mu > 1:
            raise ValueError('The Binomial parameter mu has to be in [0, 1]!')
        if mu == 1: mu = 1 - 1e-13
        if mu == 0: mu = 1e-13
        self.mu = mu # sigma latent variable a prior mean

        ''' Community-related inputs '''
        if structure not in ['assortative', 'disassortative', 'core-periphery', 'directed-biased']:
            raise ValueError('The available structures for the affinity matrix w '
                             'are: assortative, disassortative, core-periphery '
                             'and directed-biased!')
        self.structure = structure # the affinity matrix structure
        self.K = K # number of communities
        if eta <= 0 and L1:
            raise ValueError('The Dirichlet parameter eta has to be positive!')
        self.eta = eta # eta parameter of the  Dirichlet distribution
        if ag <= 0 and not L1:
            raise ValueError('The Gamma parameter alpha has to be positive!')
        self.ag = ag # alpha parameter of the Gamma distribution
        if bg <= 0 and not L1:
            raise ValueError('The Gamma parameter beta has to be positive!')
        self.bg = bg # beta parameter of the Gamma distribution
        self.L1 = L1 # flag for soft u,v generation preference, True -> Dirichlet, False -> Gamma
        if (corr < 0) or (corr > 1):
            raise ValueError('The correlation parameter has to be in [0, 1]!')
        self.corr = corr # correlation between u and v synthetically generated
        if (over < 0) or (over > 1):
            raise ValueError('The overlapping parameter has to be in [0, 1]!')
        self.over = over # fraction of nodes with mixed membership

        ''' Ranking-related inputs '''
        self.use_leagues = use_leagues
        if not self.use_leagues:
            l = 1
        self.l = l # the number of Gaussian for s
        if len(means) == self.l:
            self.means = means # means for s
        else:
            self.means = None
        if len(stds) == self.l:
            self.stds = stds # standard deviations for s
        else:
            self.stds = None
        self.beta = beta # inverse temperature parameter
        if gamma <= 0:
            raise ValueError('The spring constant gamma has to be positive!')
        self.gamma = gamma # spring constant for (s, origin)

    def EitherOr_planted_network(self, parameters=None):
        """
            Generate a directed, possibly weighted network by using the XOR model.
            Steps:
                1. Generate or load the latent variables.
                2. Extract A_ij entries (network edges) from a combination of Poisson
                   distributions;

            Parameters
            ----------
            parameters : object
                         Latent variables z, s, u, v and w.
            Returns
            ----------
            G : Digraph
                DiGraph NetworkX object. Self-loops allowed.
        """

        # Set seed random number generator
        prng = np.random.RandomState(self.prng)

        ''' Latent variables '''
        if parameters is None:
            # Generate latent variables
            self.z, self.s, self.u, self.v, self.w, nodes_s = self._generate_lv(prng)
        else:
            # Set latent variables
            self.z, self.s, self.u, self.v, self.w, nodes_s = parameters

        k_sr, k_mt, c, eps = 0., 0., 0., 0.

        if (self.z == 0).all():
            warnings.warn('All Z entries are 0: Generation with MT model.')
            self.s = np.zeros(self.N)
            S = np.zeros((self.N, self.N))
            k_mt = self.avg_degree
        else:
            # Compute normalization term for c_sr
            deltas = delta_scores(self.s)
            expH = np.exp(-self.beta * 0.5 * np.power((deltas - 1), 2))
            # Compute c_sr
            eps = 2 * self.mu * (1-self.mu) * self.delta0 * self.N
            k_sr = self.mu * (self.avg_degree - eps) * (self.mu**2 + (1-self.mu)**2)
            c = self.N * k_sr / (self.mu * (self.mu**2 + (1-self.mu)**2) * expH.sum())
            S = c * expH

        if (self.z == 1).all():
            warnings.warn('All Z entries are 1: Generation with SR model.')
            self.u = np.zeros((self.N, self.K))
            self.v = np.zeros((self.N, self.K))
            self.w = np.zeros((self.K, self.K))
            M = np.zeros((self.N, self.N))
        else:
            # Compute normalization term for c_mt
            M = np.einsum('ik,jq->ijkq', self.u, self.v)
            M = np.einsum('ijkq,kq->ij', M, self.w)
            # Update w with c_mt
            k_mt = self.avg_degree - k_sr - eps
            c_mt = self.N * k_mt / ((1 - self.mu) * (self.mu**2 + (1-self.mu)**2) * M).sum()
            self.w *= c_mt
            M *= c_mt

        ''' Network generation '''
        edge_type = self.z.copy(); edge_type[edge_type == 0] = -1 # sigma_i
        edge_mask = np.outer(edge_type, edge_type) # 1: high prob, -1: low prob (sigma_i * sigma_j)
        model_id = np.vstack([self.z] * self.N) # 0: MT, 1: SR, to be used only for coherent couples (Z_ij)

        ingroup = prng.poisson(model_id * S + (1 - model_id) * M, (self.N, self.N))
        outgroup = prng.poisson(self.delta0 * np.ones((self.N, self.N)), (self.N, self.N))

        A = np.where(edge_mask == -1, outgroup, ingroup)
        G = nx.from_numpy_matrix(A, create_using = nx.DiGraph)
        Z = np.where(A > 0, model_id, 0.5)

        ''' Network post-processing '''
        totM = np.sum(A)
        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_matrix(G, nodelist = nodes, weight = 'weight')

        # Keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key = len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        nodes = list(G.nodes())
        self.z = self.z[nodes]
        self.s = self.s[nodes]
        self.u = self.u[nodes]
        self.v = self.v[nodes]
        self.N = len(nodes)
        S = np.take(S, nodes, 1)
        S = np.take(S, nodes, 0)
        M = np.take(M, nodes, 1)
        M = np.take(M, nodes, 0)
        model_id = np.take(model_id, nodes, 1)
        model_id = np.take(model_id, nodes, 0)
        edge_mask = np.take(edge_mask, nodes, 1)
        edge_mask = np.take(edge_mask, nodes, 0)


        if self.verbose > 0:
            avg_w_deg = np.round(totM / float(G.number_of_nodes()), 3)
            avg_deg = np.round(G.number_of_edges() / float(G.number_of_nodes()), 3)

            print(f'Number of links in the upper triangular matrix: {sparse.triu(A, k = 1).nnz}\n'
                  f'Number of links in the lower triangular matrix: {sparse.tril(A, k = -1).nnz}')
            print(f'Sum of weights in the upper triangular matrix: {np.round(sparse.triu(A, k = 1).sum(), 2)}\n'
                  f'Sum of weights in the lower triangular matrix: {np.round(sparse.tril(A, k = -1).sum(), 2)}')
            print(f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component')
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                  f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (E/N): {avg_deg}')
            print(f'Average weighted degree (M/N): {avg_w_deg}')

        # Sparsity coefficient (decimal)
        sparsity_coef = lambda x: sum(x.flatten() == 0) / x.flatten().shape[0]
        if self.verbose == 2:
            print(f'Ratio: {self.mu}')
            print(f'z sparsity (~ 1 - Ratio) (%): {sparsity_coef(self.z) * 100}')
            print(f'M sparsity (%): {sparsity_coef(M) * 100}')
            print(f'S sparsity (%): {sparsity_coef(S) * 100}')
            print(f'c: {c}')
            print(f'L1: {self.L1}')

        if self.output_parameters:
            self._output_results(nodes, nodes_s, k_sr, k_mt, c, edge_mask, model_id)

        if self.output_adj:
            self._output_adjacency(G, outfile = self.outfile_adj)

        return G

    def _generate_lv(self, prng=None):
        """
            Generate z, s, u, v, w latent variables.
            Parameters
            ----------
            prng : random generator container
                   Seed for the random number generator.
            Returns
            ----------
            z : Numpy array
                Matrix NxN of model indicators (binary).

            s : Numpy array
                N-dimensional array of real ranking scores for each node.

            u : Numpy array
                Matrix NxK of out-going membership vectors, positive element-wise.
                With unitary L1 norm computed row-wise.

            v : Numpy array
                Matrix NxK of in-coming membership vectors, positive element-wise.
                With unitary L1 norm computed row-wise.

            w : Numpy array
                Affinity matrix KxK. Possibly None if in pure SpringRank.
                Element (k,h) gives the density of edges going from the nodes
                of group k to nodes of group h.

            nodes_s : Numpy array
                      Result of the random permutation applied to the node IDs (if required).
                      Can be used for inverting the permutation and induce the block structure
                      generated by the leagues on the adjacency matrix.
        """
        if prng is None:
            # Set seed random number generator
            prng = np.random.RandomState(seed = 42)

        # Generate z through binomial distribution
        z = prng.binomial(1, self.mu, self.N)
        # Generate s through gaussians
        s, nodes_s = ranking_scores(prng, self.use_leagues, self.permute, self.gamma, self.beta, self.N, self.l,
                                    self.means, self.stds)
        # Generate u,v,w for possibly overlapping communities
        u, v = membership_vectors(prng, self.L1, self.eta, self.ag, self.bg, self.K, self.N, self.corr, self.over)
        w = affinity_matrix(self.structure, self.N, self.K, self.avg_degree)

        return z, s, u, v, w, nodes_s

    def _output_results(self, nodes, nodes_s, k_sr, k_mt, c, edge_mask, model_id):
        """
            Output results in a compressed file.
            Parameters
            ----------
            nodes : list
                    List of nodes IDs.
            nodes_s: Numpy array
                     Result of the random permutation applied to the node IDs (if required)
            k_sr : float
                   Fraction of average degree given by the hierarchical mechanism.
            k_mt : float
                   Fraction of average degree given by the community mechanism.
            c : float
                Overall sparsity coefficient.
            edge_mask : Numpy array
                        Mask that return the adjacency matrix entries representing in-group connections.
            model_id : Numpy array
                       Node type vector.

        """

        output_parameters = self.folder + 'results_' + self.label + '_' + str(self.prng)
        np.savez_compressed(output_parameters + '.npz', s = self.s, u = self.u, v = self.v,
                            w = self.w, z = model_id, edge_mask = edge_mask, mu = self.mu, beta = self.beta,
                            sigma = self.z, delta0 = self.delta0, k_sr = k_sr, k_mt = k_mt, c = c, nodes = nodes,
                            nodes_s = nodes_s)
        if self.verbose:
            print()
            print(f'Parameters saved in: {output_parameters}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _output_adjacency(self, G, outfile=None):
        """
            Output the adjacency matrix. Default format is space-separated .csv
            with 3 columns: node1 node2 weight

            Parameters
            ----------
            G: Digraph
               DiGraph NetworkX object.
            outfile: str
                     Name of the adjacency matrix.
        """

        if outfile is None:
            outfile = 'syn_' + self.label + '_' + str(self.prng) + '.dat'

        edges = list(G.edges(data = True))
        try:
            data = [[u, v, d['weight']] for u, v, d in edges]
        except:
            data = [[u, v, 1] for u, v, d in edges]

        df = pd.DataFrame(data, columns = ['source', 'target', 'w'], index = None)
        df.to_csv(self.folder + outfile, index = False, sep = ' ')
        if self.verbose:
            print(f'Adjacency matrix saved in: {self.folder + outfile}')


def ranking_scores(prng=None, mix=False, permute=False, gamma=0.01, beta=5., N=100, l=1, means=None, stds=None):
    """
        Generate the ranking scores.

        Parameters
        ----------
        prng : random generator container
               Seed for the random number generator.
        mix : bool
              Flag for generating the ranking scores with a Gaussian mixture.
        permute : bool
                  Flag for permuting the node before associating a ranking score to each of them,
                  i.e. the hierarchical block structure induced on the adjacency matrix is randomized.
        gamma : float
                The spring constant for (s, origin).
        beta : float
               Inveres temperature parameter.
        N : int
            Number of nodes.
        l : int
            Number of leagues
        means : list
                List of means to be used for the scores generation.
        stds : list
               List of means to be used for the scores generation.

        Returns
        ----------
        s : Numpy array
            N-dimensional array of real ranking scores for each node.

        nodes_s : Numpy array
                  Result of the random permutation applied to the node IDs (if required).
                  Can be used for inverting the permutation and induce the block structure
                  generated by the leagues on the adjacency matrix.
    """
    if prng is None:
        # Set seed random number generator
        prng = np.random.RandomState(seed = 42)
    if mix:
        if means is None:
            means = prng.randint(-5, 5, l)
        if stds is None:
            stds = prng.randint(0, 1, l)
        s = np.concatenate([prng.normal(means[i], stds[i], N // l) for i in range(l - 1)])
        if N % l:
            s = np.concatenate([s, prng.normal(means[-1], stds[-1], N - s.shape[0])])
        if permute:
            # shuffle s in order to not have a ranking structure overlapped to the communities one
            nodes_s = prng.permutation(N)
            s = s[nodes_s]
        else:
            nodes_s = np.arange(N)
    else:
        # Generate s through factorized Gaussian, l0 = 0
        s = prng.normal(0, 1. / np.sqrt(gamma * beta), N)
        nodes_s = np.arange(N)

    return s, nodes_s


def membership_vectors(prng=None, L1=False, eta=0.5, alpha=0.6, beta=1, K=3, N=100, corr=0., over=0.):
    """
        Compute the NxK membership vectors u, v using a Dirichlet or a Gamma distribution.
        Parameters
        ----------
        prng: Numpy Random object
              Random number generator container.
        L1 : bool
             Flag for parameter generation method. True for Dirichlet, False for Gamma.
        eta : float
              Parameter for Dirichlet.
        alpha : float
            Parameter (alpha) for Gamma.
        beta : float
            Parameter (beta) for Gamma.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        corr : float
               Correlation between u and v synthetically generated.
        over : float
               Fraction of nodes with mixed membership.
        Returns
        -------
        u : Numpy array
            Matrix NxK of out-going membership vectors, positive element-wise.
            Possibly None if in pure SpringRank or pure MultiTensor.
            With unitary L1 norm computed row-wise.

        v : Numpy array
            Matrix NxK of in-coming membership vectors, positive element-wise.
            Possibly None if in pure SpringRank or pure MultiTensor.
            With unitary L1 norm computed row-wise.
    """
    if prng is None:
        # Set seed random number generator
        prng = np.random.RandomState(seed = 42)
    # Generate equal-size unmixed group membership
    size = int(N / K)
    u = np.zeros((N, K))
    v = np.zeros((N, K))
    for i in range(N):
        q = int(math.floor(float(i) / float(size)))
        if q == K:
            u[i:, K - 1] = 1.
            v[i:, K - 1] = 1.
        else:
            for j in range(q * size, q * size + size):
                u[j, q] = 1.
                v[j, q] = 1.
    # Generate mixed communities if requested
    if over != 0.:
        overlapping = int(N * over)  # number of nodes belonging to more than 1 communities
        ind_over = np.random.randint(len(u), size = overlapping)
        if L1:
            u[ind_over] = prng.dirichlet(eta * np.ones(K), overlapping)
            v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.dirichlet(eta * np.ones(K), overlapping)
            if corr == 1.:
                assert np.allclose(u, v)
            if corr > 0:
                v = normalize_nonzero_membership(v)
        else:
            u[ind_over] = prng.gamma(alpha, 1. / beta, size = (N, K))
            v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.gamma(alpha, 1. / beta, size = (overlapping, K))
            u = normalize_nonzero_membership(u)
            v = normalize_nonzero_membership(v)
    return u, v


def affinity_matrix(structure='assortative', N=100, K=3, a=0.1, b=0.5):
    """
        Compute the KxK affinity matrix w with probabilities between and within groups.
        Parameters
        ----------
        structure : string
                    Structure of the network.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        a : float
            Parameter for secondary probabilities.
        b : float
            Parameter for secondary probabilities.
        Returns
        -------
        p : Numpy array
            Array with probabilities between and within groups. Element (k,h)
            gives the density of edges going from the nodes of group k to nodes of group h.
    """

    b *= a
    p1 = K / N
    while p1 < a:
        a *= 0.1

    if structure == 'assortative':
        p = p1 * a * np.ones((K, K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

    elif structure == 'disassortative':
        p = p1 / K * np.ones((K, K))  # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    elif structure == 'core-periphery':
        p = p1 / K * np.ones((K, K))
        np.fill_diagonal(np.fliplr(p), a * p1)
        p[1, 1] = b * p1

    elif structure == 'directed-biased':
        p = a * p1 * np.ones((K, K))
        p[0, 1] = p1
        p[1, 0] = b * p1

    return p


def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.
        Parameters
        ----------
        u: Numpy array
           Numpy Matrix.
        Returns
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis = 1, keepdims = True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1


@jit(nopython = True)
def delta_scores(s):
    """
        Compute the pairwise ranking differences.
    """
    N = s.shape[0]
    delta_s = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            delta_s[i, j] = s[i] - s[j]
    return delta_s
