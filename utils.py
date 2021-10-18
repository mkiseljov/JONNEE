import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt


## Miscellaneous useful functions ##


def load_graph_to_numpy(path_to_edgelist):
    pass


def load_graph(path_to_edgelist, subgraph=None, weighted=False, seed=0):
    """
    :param subgraph: None or int - number of nodes to sample
    """
    # nx_graph = nx.from_edgelist([l.split() for l in open(path_to_edgelist)])
    if weighted:
        nx_graph = nx.read_weighted_edgelist(path_to_edgelist)
    else:
        nx_graph = nx.read_edgelist(path_to_edgelist)
    print('The graph has {} nodes and {} edges'.format(nx_graph.number_of_nodes(),
                                                       nx_graph.number_of_edges()))
    if subgraph is None:
        return nx_graph

    if seed:
        np.random.seed(seed)
    print('Sampling {}-node subgraph from original graph'.format(subgraph))
    return nx_graph.subgraph(np.random.choice(nx_graph.nodes(),
                            size=subgraph, replace=False))

def get_dual(graph, sparse=True):
    # graph is a networkx Graph
    L = nx.line_graph(graph)
    nodelist = sorted(L.nodes())
    # may wrap sp.csr around numpy
    if sparse:
        return nx.to_scipy_sparse_matrix(L, nodelist), nodelist
    return nx.to_numpy_matrix(L, nodelist), nodelist


# def get_dual(adj):
#     # adj is a networkx Graph
#     adj = nx.from_numpy_array(adj)
#     L = nx.line_graph(adj)
#     nodelist = sorted(L.nodes())
#     return nx.to_numpy_matrix(L, nodelist), {i: n for i, n in enumerate(nodelist)}


def get_features(adj, sparse=True):
    if sparse:
        return sp.eye(adj.shape[0])
    return np.identity(adj.shape[0])



###############  VGAE-specific  ################
########################################################################################

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and 
# https://github.com/tkipf/gae
# ------------------------------------

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae_lp(edges_pos, edges_neg, emb, adj_orig, threshold=0.5, verbose=False):
    """
    Evaluate VGAE learned embeddings on Link Prediction task.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    emb = emb.data.numpy()
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []

    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []

    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    
    if verbose:
        print("EVAL GAE")
        p = np.random.choice(range(len(preds_all)), replace=False, size=min([len(preds_all), 100]))
        print(preds_all[p])
        print(labels_all[p])
    
    accuracy = accuracy_score((preds_all > threshold).astype(float), labels_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1score = f1_score(labels_all, (preds_all > threshold).astype(float))
    logloss = log_loss(labels_all, preds_all)

    return accuracy, roc_score, ap_score, f1score, logloss


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(
            pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'),  encoding='latin1'))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
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
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


# Subsample sparse variables
def get_subsampler(variable):
    data = variable.view(1, -1).data.numpy()
    edges = np.where(data == 1)[1]
    nonedges = np.where(data == 0)[1]

    def sampler():
        idx = np.random.choice(
            nonedges.shape[0], edges.shape[0], replace=False)
        return torch.LongTensor(np.append(edges, nonedges[idx]))
    
    return sampler


def plot_results(results, test_freq, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'])
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss (ELBO)')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results['accuracy_train'])
    ax.plot(x_axis_test, results['accuracy_test'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    plt.show()
    # fig.savefig(path)


def load_multiclass(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if 'cora' in path:
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
        
        # idx_train = range(140)
        # idx_val = range(200, 500)
        # idx_test = range(500, 1500)
        rng = list(range(1500))
        np.random.shuffle(rng)
        idx_train = rng[:1000]
        idx_val = rng[1000:1200]
        idx_test = rng[1200:1500]
        # idx_train = np.random.choice(range(1500), replace=False, size=1000)
        # idx_val = np.random.choice(range(1500), replace=False, size=300)
        # idx_test = np.random.choice(range(1500), replace=False, size=200)
        labels = torch.LongTensor(np.where(labels)[1])

    elif 'Blog' in path:
        labels = np.array([np.random.choice(
                            np.where(
                                np.array( list(map(int, line.split())) )
                                )[0],
                            size=1)
                            for line in open(path+'blog_catalog.classes')],
                            dtype='int')
        features = sp.eye(labels.shape[0])
        idx = np.arange(labels.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array([list(map(int, line.split())) 
                            for line in open(path+'blog_catalog.edgelist')])
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        
        n = labels.shape[0]
        idx_train = range(int(n * 0.6))
        idx_val = range(int(n * 0.6), int(n * 0.8))
        idx_test = range(int(n * 0.8), n)
        # idx_train = np.random.choice(range(n), replace=False, size=int(n * 0.6))
        # idx_val = np.random.choice(range(n), replace=False, size=int(n * 0.2))
        # idx_test = np.random.choice(range(n), replace=False, size=int(n * 0.2))
        labels = torch.LongTensor(labels)
        labels = labels.view(labels.size(0))

    # TODO: other datasets here

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#####################################################
# Jupyter utils #
#####################################################

def log_progress(sequence, every=10):
    # thanks go to:
    # https://habrahabr.ru/post/276725/
    from ipywidgets import IntProgress
    from IPython.display import display

    progress = IntProgress(min=0, max=len(sequence), value=0)
    display(progress)
    
    for index, record in enumerate(sequence):
        if index % every == 0:
            progress.value = index
        yield record


