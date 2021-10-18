# PyTorch model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
from pyro.distributions import normal
from pyro.util import ng_zeros, ng_ones

from DuoGAE.layers import GraphConvolution
from DuoGAE.distributions import weighted_bernoulli
from DuoGAE.distributions import VonMisesFisher
from DuoGAE.utils import make_sparse

from DuoGAE.preprocessing import preprocess_train_test_split

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score,\
                            average_precision_score,\
                            accuracy_score,\
                            f1_score,\
                            log_loss


class GCNEncoder(nn.Module):
    """
    Encoder using GCN layers
    """

    def __init__(self, n_feat, n_hid, n_latent, dropout):
        """
        TODO: add batchnorm / dropout as in
        https://github.com/3ammor/Variational-Autoencoder-pytorch/tree/master/graph
        """
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2_mu = GraphConvolution(n_hid, n_latent)
        self.gc2_sig = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout


    def forward(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        log_sig = self.gc2_sig(x, adj)
        return mu, torch.exp(log_sig)


class GCNEncoderSimple(nn.Module):
    """
    Encoder using GCN layers
    """

    def __init__(self, n_feat, n_hid, n_latent, dropout):
        super(GCNEncoderSimple, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        return x


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, adj_mat):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7

    def forward(self, z):
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj


class InnerProductDecoderMasked(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, adj_mat, beta=0.):
        super(InnerProductDecoderMasked, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7
        self.mask = Variable(torch.FloatTensor(adj_mat.data.to_dense().numpy() * beta + 1))

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t()) * self.mask) + self.fudge) * (1 - 2 * self.fudge)
        return adj



class GAE(object):
    """Graph Auto Encoder (see: https://arxqiv.org/abs/1611.07308)"""

    def __init__(self, data, n_hidden, n_latent, dropout, mask=1e-2):
        super(GAE, self).__init__()
        
        N, D = data['features'].shape

        # Data
        self.x = data['features']
        self.adj_norm   = data['adj_norm']
        self.adj_labels = data['adj_labels']
        self.L_norm = Variable(make_sparse(sp.eye(N)).sub(self.adj_norm.data))

        # Dimensions
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.mask = mask

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)

        # Layers 
        self.dropout = dropout
        self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
        self.decoder = InnerProductDecoderMasked(self.dropout, self.adj_norm, self.mask)


    def model(self):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # Setup hyperparameters for prior p(z)
        z_mu = ng_zeros([self.n_samples, self.n_latent])
        z_sigma = ng_ones([self.n_samples, self.n_latent])
        
        # sample from prior
        z = pyro.sample("latent", normal, z_mu, z_sigma)  # TODO: vonMisesFisher prior

        # decode the latent code z
        z_adj = self.decoder(z)

        # Reweighting
        with pyro.iarange("data"):
            pyro.observe('obs', weighted_bernoulli, self.adj_labels.view(1, -1),
                         z_adj.view(1, -1), weight=self.pos_weight)


    def guide(self):
        # register PyTorch model 'encoder' w/ pyro
        pyro.module("encoder", self.encoder)

        # Use the encoder to get the parameters use to define q(z|x)
        z_mu, z_sigma = self.encoder(self.x, self.adj_norm)

        # Sample the latent code z
        pyro.sample("latent", normal, z_mu, z_sigma)  # TODO: vonMisesFisher posterior


    def get_embeddings(self):
        """
        Returns shape N x d
        (self.n_samples, self.n_latent)
        """
        z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
        self.encoder.train()  # Put encoder back into training mode
        return z_mu


###########################################################################
########################          GAE         #############################
###########################################################################



class GAE_NP(nn.Module):
    """
    Ordinary (non-probabilistic) Graph Auto Encoder

    Allows for hard beta-masking

    """

    def __init__(self, data, n_hidden, n_latent, dropout):
        super(GAE_NP, self).__init__()
        
        N, D = data['features'].shape

        # Data
        self.x = data['features']
        self.adj_norm   = data['adj_norm']
        self.adj_labels = data['adj_labels']
        self.L_norm = Variable(make_sparse(sp.eye(N)).sub(self.adj_norm.data))

        # Dimensions
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Layers 
        self.dropout = dropout
        self.encoder = GCNEncoderSimple(self.input_dim, self.n_hidden,
                                        self.n_latent, self.dropout)
        self.decoder = InnerProductDecoder(self.dropout, self.adj_norm)


    def forward(self, x, adj):
        z = self.encoder(x, adj)
        adj_hat = self.decoder(z)
        return adj_hat


    def get_embeddings(self):
        """
        Returns shape N x d
        (self.n_samples, self.n_latent)
        """
        z = self.encoder(self.x, self.adj_norm)
        return z




###########################################################################
########################  Evaluation of VGAE  #############################
###########################################################################


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


def eval_gae_mc():
    raise NotImplementedError


