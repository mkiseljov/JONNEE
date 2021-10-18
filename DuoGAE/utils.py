# utilities

import numpy as np
import networkx as nx
import random
import argparse
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt

# from DuoGAE.preprocessing import preprocess_file

#################### Data loaders ######################

def load_graph(path, subgraph=None, weighted=False, seed=0,
               subnodes_dict=None):
    """
    :param subgraph: None or int - number of nodes to sample
    """
    # nx_graph = nx.from_edgelist([l.split() for l in open(path_to_edgelist)])
    path_to_edgelist = path + ".preprocessed"
    preprocess_file(frm=path, to=path_to_edgelist)
    if weighted: nx_graph = nx.read_weighted_edgelist(path_to_edgelist)
    else: nx_graph = nx.read_edgelist(path_to_edgelist)

    if subgraph is None:
        print('The graph has {} nodes and {} edges'.format(nx_graph.number_of_nodes(),
                                                           nx_graph.number_of_edges()))
        return nx_graph

    # else: we need to sample a subgraph
    np.random.seed(seed)
    print('Sampling a {}-node subgraph from original graph'.format(subgraph))
    
    # !! Problem: this graph is almost certainly disconnected !!
    subgraph_nodes = np.random.choice(nx_graph.nodes(),
                                      size=subgraph,
                                      replace=False)

    if subnodes_dict is not None:
        subnodes_dict['nodes'] = subgraph_nodes

    graph = nx_graph.subgraph(subgraph_nodes)
    # print(len(graph.nodes()))
    # edgelist = sg.edges()
    # ndmap = {n: i for i, n in enumerate(sg.nodes())}
    # return nx.parse_edgelist([ (ndmap[e[0]], ndmap[e[1]]) for e in edgelist])
    delim = " "
    with open(path_to_edgelist, 'w') as f_to:
        # num_nodes = graph.number_of_nodes()
        mapper = {n: i for i, n in enumerate(sorted(graph.nodes()))}
        
        for edge in graph.edges():
            line = str(mapper[edge[0]]) + delim + str(mapper[edge[1]])
            if weighted:
                line += delim + str(graph[edge[0]][edge[1]]['weight'])
            f_to.write(line + "\n")

        # add self-connections
        for i in range(graph.number_of_nodes()):
            f_to.write(str(i) + delim + str(i) + "\n")

    if weighted:
        return nx.read_weighted_edgelist(path_to_edgelist)
    return nx.read_edgelist(path_to_edgelist)



def get_dual(graph, sparse=True):
    # graph is a networkx Graph
    L = nx.line_graph(graph)
    nodelist = sorted(L.nodes())
    # may wrap sp.csr around numpy
    if sparse:
        return nx.to_scipy_sparse_matrix(L, nodelist), nodelist
    return nx.to_numpy_matrix(L, nodelist), nodelist


######################################################


class SequenceLearner(object):
    def __init__(self, args):
        raise NotImplementedError

    def learn_embeddings(self, nx_G):
        """
        dataset - GraphDataset instance
        nx_G - networkx graph (training)
        """
        raise NotImplementedError

    def get_embeddings(self):
        return self.embeddings

    def get_params(self):
        return self.__dict__

    def _learn_from_sequences(self, walks):
        walks = [list(map(str, walk)) for walk in walks]
        if self.name == 'diff2vec':
            model = Word2Vec(walks,
                         vector_size=self.dimensions,
                         window=self.window_size,
                         min_count=1, sg=1,
                         epochs=self.iter,
                         alpha=self.alpha)
        else:
            model = Word2Vec(walks,
                         vector_size=self.dimensions,
                         window=self.window_size,
                         min_count=0, sg=1,
                         epochs=self.iter)

        # model.wv.save_word2vec_format(self.output)
        return self._from_wv(model.wv)


    def _from_wv(self, emb_wv):
        emb = np.zeros(shape=(self.n, self.dimensions))
        i2w = emb_wv.index_to_key
        # print(len(i2w))
        for w in i2w:
            emb[int(w), :] = emb_wv[w]
        return emb


######## Pytorch conversions ##########

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


######## Other utilities ########


def preprocess_file(frm, to, delim=" "):
    with open(to, 'w') as f_to:
        with open(frm, 'r') as f_frm:
            line = f_frm.readline()
            l = len(line.split())
            weighted = (l == 3)
        if weighted:
            graph = nx.read_weighted_edgelist(frm, delimiter=delim)
        else:
            graph = nx.read_edgelist(frm)
        num_nodes = graph.number_of_nodes()
        mapper = {n: i for i, n in enumerate(sorted(graph.nodes))}
        for edge in graph.edges():
            line = str(mapper[edge[0]]) + delim + str(mapper[edge[1]])
            if weighted:
                line += delim + str(graph[edge[0]][edge[1]]['weight'])
            f_to.write(line + "\n")




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



