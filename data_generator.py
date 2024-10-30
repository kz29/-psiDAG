import networkx as nx
import igraph as ig
import numpy as np
import torch
import random




# Sythentic dataset

class generate_DAG:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """

    def __init__(self, d, graph_type, degree, B_ranges, hd: bool = False, seed=0):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            seed (int): Random seed. Default: 1.
            
        """
        
        torch.backends.cudnn.deterministic = True
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.B_ranges = B_ranges
        self.rs = np.random.RandomState(seed)  # Reproducibility
        self.hd = hd
        self._setup()
        


        
    def _setup(self):
        """Generate B_bin, B and X."""
        self.B_bin = self.simulate_random_dag(d=self.d, degree=self.degree,
                                              graph_type=self.graph_type, rs=self.rs)
        self.B = self.simulate_weight(B_bin=self.B_bin, B_ranges=self.B_ranges, hd=self.hd, rs=self.rs)

    def simulate_er_dag(self, d, degree, rs):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """

        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_array(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)  # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin


    def simulate_sf_dag(self, d, degree, rs):
        """Simulate SF DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin
    
    def simulate_random_dag(self, d, degree, graph_type, rs):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """

        def _random_permutation(B_bin):
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = self.simulate_er_dag(d, degree, rs)
            
        elif graph_type == 'SF':
            B_bin = self.simulate_sf_dag(d, degree, rs)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)


    # new weights for the dataset
    def simulate_weight(self, B_bin, B_ranges, hd, rs):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        # Which range
        S = rs.randint(len(B_ranges), size=B.shape)  
        for i, (low, high) in enumerate(B_ranges):
            if hd == True:

                if np.random.rand(1) > 0.5:
                    U = rs.uniform(low=low, high=high, size=B.shape)
                else:
                    U_0 = rs.uniform(low=-3, high=-1, size=B.shape)
                    U = rs.uniform(low=low, high=high, size=B.shape) * np.exp(U_0)
                B += B_bin * (S == i) * U

            else:
                U = rs.uniform(low=low, high=high, size=B.shape)
                B += B_bin * (S == i) * U

        return B
    
def set_random_seed(seed: int):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generator_matrix(B):
    return torch.linalg.inv(torch.diag(torch.ones_like(B[0])) - B)



def data_generator(generator_matrix, bs=1, noise_type='gaussian_ev', seed=0):
    
    d = torch.tensor(generator_matrix[0].size())[0]
    
    if noise_type == 'gaussian_ev':
        # Gaussian noise with equal variances
        N_i = torch.randn(bs, d).type_as(generator_matrix[0])

    elif noise_type == 'exponential':
        exponential_dist = torch.distributions.Exponential(1.0)
        # Exponential noise
        
        N_i = exponential_dist.sample((bs, d)).type_as(generator_matrix[0])
        
    elif noise_type == 'gumbel':
        # Gumbel noise
       gumbel_dist = torch.distributions.Gumbel(0.0, 1.0)
       N_i = gumbel_dist.sample((bs, d)).type_as(generator_matrix[0])
    else:
        raise ValueError("Unknown noise type.")
    return N_i @ generator_matrix
