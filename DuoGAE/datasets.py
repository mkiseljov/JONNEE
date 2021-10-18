# per class for each dataset:
# BlogCatalog
# HepTh
# Astrophys
# PPI

from copy import copy
from DuoGAE.utils import load_graph, get_dual
import networkx as nx
import scipy.sparse as sp
import numpy as np
from DuoGAE.preprocessing import preprocess_train_test_split
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle as pkl
import sys

"""

load_graph() does the renaming

"""


class GraphDataset(object):
    def __init__(self, nx_G, features=None, labels=None):
        self.nx_G = nx_G
        self.n = nx_G.number_of_nodes()
        self.nodelist = sorted(nx_G.nodes())
        self.edgelist = sorted(nx_G.edges())  # TODO: check this
        self.adj = sp.csr_matrix(nx.to_numpy_matrix(nx_G, self.nodelist))
        self.features = features
        self.labels = labels  # -- for semisupervised
        self.node2ind_map = {node: i for i, node in enumerate(self.nodelist)}

        ## Boolean flags:
        self.has_features = False if features is None else True
        self.weighted = False
        self.noeval = False


    def get_data(self):
        data = {
                'adj':  copy(self.adj),
                'features': copy(self.features)
                }
        return data

    def get_nx_graph(self):
        return self.nx_G

    def preprocess_train_test_split(self, seed, train_test_split=(0.85, 0.05, 0.1)):
        """
        self.t is a dict with fields:

        'adj_train', 'adj_train_norm',
        'adj_train_labels', 'train_edges',
        'val_edges', 'val_edges_false',
        'test_edges', 'test_edges_false'

        """
        self.t = preprocess_train_test_split(self.adj, seed,
                    proportions=train_test_split)

    def get_train_data(self):
        # to be called after a .train_test_split() call
        pass

    def get_test_data(self):
        pass

    def make_weighted(self):
        """
        Add dummy weights unless graph is already weighted
        """
        if not self.weighted:
            for edge in self.edgelist:
                self.nx_G[edge[0]][edge[1]]['weight'] = 1.0


def to_dual(adj, sparse=True):
    """
    Instantiate and return a dual Dataset object
    """
    nx_G = nx.from_scipy_sparse_matrix(adj)
    adj_dual, nodes_dual = get_dual(nx_G, sparse)
    if sparse:
        nx_G_dual = nx.from_scipy_sparse_matrix(adj_dual)
    else:
        nx_G_dual = nx.from_numpy_matrix(adj_dual)
    dataset_dual = GraphDataset(nx_G_dual)
    # print("Dual adjacency matrix has shape", adj_dual.shape)
    return dataset_dual

######## Concrete implementations #######

class HepThDataset(GraphDataset):
    """
    Cit-HepTh Dataset
    A large citation network
    """
    def __init__(self, size=500, seed=10,
        path='../data/cit-hepth/cit-HepTh.edgelist'):
        nx_G = load_graph(path,
                          weighted=False,
                          subgraph=size,
                          seed=seed)
        super(HepThDataset, self).__init__(nx_G)


class AstroPhDataset(GraphDataset):
    """
    Collaboration network
    5k nodes
    """
    def __init__(self, size=500, seed=10,
        path='../data/ca-grqc/ca-GrQc.edgelist'):
        nx_G = load_graph(path,
                          weighted=False,
                          subgraph=size,
                          seed=seed)
        super(AstroPhDataset, self).__init__(nx_G)



class FacebookDataset(GraphDataset):
    """
    Facebook dataset
    Additionally has (?) circles attached
    """
    def __init__(self, size=500, seed=10,
        path='../data/facebook/facebook_combined.txt'):
        nx_G = load_graph(path,
                  weighted=False,
                  subgraph=size,
                  seed=seed)
        super(FacebookDataset, self).__init__(nx_G)


#### Multiclass datasets ####

class BlogCatalogDataset(GraphDataset):
    """
    Large and multilabel
    """
    def __init__(self, size=500, seed=10,
        path='../data/BlogCatalog-dataset/data/blog_catalog.edgelist'):
        # load dataset
        # also do the renaming
        dct = {}
        nx_G = load_graph(path,
                          weighted=False,
                          subgraph=size,
                          seed=seed,
                          subnodes_dict=dct)

        class_path = '../data/BlogCatalog-dataset/data/blog_catalog.classes'
        nds = list(map(int, dct['nodes']))
        labels = np.array([list(map(int, line.split()))
                           for line in open(class_path)],
                           dtype='int')
        # print(labels.shape)
        labels = labels[nds, :]
        super(BlogCatalogDataset, self).__init__(nx_G, labels=labels)

#### Datasets with features ####

class PPIDataset(GraphDataset):
    """
    Large and multilabel
    Has node features
    """
    def __init__(self, size=500, seed=10,
        compress_to=None,
        path='../data/ppi/ppi.edgelist'):
        dct  = {}
        nx_G = load_graph(path,
                          weighted=False,
                          subgraph=size,
                          seed=seed,
                          subnodes_dict=dct)

        # load features from ppi-feats.npy
        node_inds = list(map(int, dct['nodes']))  # order ids of our nodes
        nx_graph = nx.read_edgelist(path)
        nodes = sorted(list(nx_graph.nodes()))
        nds = [int(n) for i, n in enumerate(nodes) if i in node_inds]
        
        sc = StandardScaler()
        X = np.load('../data/ppi/ppi-feats.npy')[nds, :]
        X = sc.fit_transform(X)
        # X = None
        print("Features have shape", X.shape)
        if compress_to is not None:
            pca = PCA(n_components=compress_to)
            X = pca.fit_transform(X)
        # TODO: load labels (no not forget to subselect)

        # this is a map: "1001" -> [1, 1, 0, 0, ... ] (total 50 entries)
        class_map = json.load(open('../data/ppi/ppi-class_map.json'))
        labels = np.array([class_map[n] for n in dct['nodes']])
        # (See notebook in ppi folder)
        super(PPIDataset, self).__init__(nx_G, features=X, labels=labels)


class CoraDataset(GraphDataset):
    """
    Multilabel 
    (but with only one label per node, so good for viz)
    Has node features
    """
    def __init__(self, path="../data/cora/", compress_to=None, **kwargs):
        # adj, features, labels = load_cora()
        # adj, features, labels = load_cora_another_function()
        # nx_G = nx.from_scipy_sparse_matrix(adj)
        # idx_features_labels = np.genfromtxt("{}{}.content".format(path, "cora"),
        #                                     dtype=np.dtype(str))
        # labels = idx_features_labels[:, -1]
        adj, features, labels = load_cora()
        print("Features have shape", features.shape)
        if compress_to is not None:
            pca = PCA(n_components=compress_to)
            features = pca.fit_transform(features)
        nx_G = nx.from_scipy_sparse_matrix(adj)
        super(CoraDataset, self).__init__(nx_G, features, labels)
        self.label_map = ['Case_Based',
                          'Genetic_Algorithms',
                          'Neural_Networks',
                          'Probabilistic_Methods',
                          'Reinforcement_Learning',
                          'Rule_Learning',
                          'Theory']

##### Weighted datasets #####

class HSEDataset(GraphDataset):
    """
    Has node weights
    """
    def __init__(self, path='../data/hse/edgelist_2017_n.txt', **kwargs):
        nx_G = load_graph(path, weighted=True)
        super(HSEDataset, self).__init__(nx_G)
        self.weighted = True



###### Visualization ###########

class KarateDataset(GraphDataset):
    """
    Has node weights
    """
    def __init__(self, path='../data/karate/karate.edgelist', **kwargs):
        nx_G = load_graph(path, weighted=False)
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                  1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
                  0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0]
        reorder = np.array(sorted(list(map(str, list(range(34))))),
                           dtype="int32")
        labels = np.array(labels)[reorder]
        super(KarateDataset, self).__init__(nx_G, labels=labels)
        self.noeval = True



########## Helpers #############

def load_cora(path="../data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                    dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)} # node_name -> number
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = np.array(features.todense())

    return adj, features, labels


def load_cora_another_function(dataset="cora"):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/cora/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "../data/cora/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features = features.todense()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, None

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
