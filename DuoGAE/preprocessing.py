# VGAE preprocessing

import numpy as np
import scipy.sparse as sp
from time import time
import networkx as nx

import torch
from torch.autograd import Variable

from DuoGAE.utils import sparse_mx_to_torch_sparse_tensor,\
                         sparse_to_tuple, make_sparse

# from DuoGAE.utils import make_sparse



def make_incidence_matrix(node_map, edges):
    """
    Make a (V x E) vertex-edge incidence matrix
    edges : dictionary,
    idx_train - training nodes
    """
    ident = isinstance(node_map, int)
    if ident:
        n = node_map
    else:
        n = len(node_map)
    row, col = [], []
    for edge_id, edge in enumerate(edges):
        a, b = edge
        if not ident:
            row.append(node_map[a]); col.append(edge_id)
            row.append(node_map[b]); col.append(edge_id)
        else:
            row.append(a); col.append(edge_id)
            row.append(b); col.append(edge_id)
    arr = sp.coo_matrix(([1] * len(row), (row, col)),
                        shape=(n, len(edges)))


    # TODO!!! : return degrees matrix as well
    return sparse_mx_to_torch_sparse_tensor(arr)


# def make_incidence_matrix(adj):
#     """
#     Make a (V x E) vertex-edge incidence matrix
#     No masking (multiclass classif)
#     """
#     edges = nx.from_scipy_sparse_matrix(adj).edges()
#     row, col = [], []
#     for edge_id, edge in enumerate(edges):
#         a, b = edge
#         col.append(a); row.append(edge_id)
#         col.append(b); row.append(edge_id)
#     arr = sp.coo_matrix(([1] * len(row), (row, col)),
#                         shape=(len(edges), adj.shape[0]))
#     return sparse_mx_to_torch_sparse_tensor(arr.transpose())



def preprocess_train_test_split(adj, seed, proportions=(0.85, 0.05, 0.1)):
    # Performs a 85-5-10 split
    t0 = time()
    adj_orig = adj
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, seed, proportions)

    # Some preprocessing
    adj_train_norm   = normalize_adj(adj_train)
    adj_train_norm   = Variable(make_sparse(adj_train_norm))
    adj_train_labels = Variable(torch.FloatTensor(adj_train + sp.eye(adj_train.shape[0]).todense()))

    print('Preprocessing time: {:.2f}s'.format(time() - t0))

    return {'adj_train': adj_train,
            'adj_train_norm': adj_train_norm,
            'adj_train_labels': adj_train_labels,
            'train_edges': train_edges,
            'val_edges': val_edges, 'val_edges_false': val_edges_false,
            'test_edges': test_edges, 'test_edges_false': test_edges_false}


def normalize_adj(adj):
    """
    Normalize the adjacency matrix as D^-1/2 A D^-1/2
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def mask_test_edges(adj, seed, proportions):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Note: adj is sparse CSR

    np.random.seed(seed)
    
    # Remove diagonal elements
    adj = adj - \
        sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * proportions[2]))
    num_val = int(np.floor(edges.shape[0] * proportions[1]))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack(
        [test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        try:
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                    np.all(np.any(rows_close, axis=0), axis=0))
        except:
            return True

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false



