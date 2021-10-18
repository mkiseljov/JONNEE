import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones
from pyro.infer import SVI
from pyro.optim import Adam

from layers import GraphConvolution, GraphConvolution_GCN, SparseMM
from utils import get_subsampler, make_sparse
from utils import eval_gae_lp, plot_results
from preprocessing import preprocess_train_test_split

from dist import weighted_bernoulli

from collections import defaultdict
from tqdm import tqdm
from utils import log_progress

########################################################################################
#                                                                                      #
#######################   GCN - graph convolutional network   ##########################
#                                                                                      #
########################################################################################

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution_GCN(nfeat, nhid)
        self.gc2 = GraphConvolution_GCN(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)





########################################################################################
#                                                                                      #
######################   VGAE - Variational graph autoencoder   ########################
#                                                                                      #
########################################################################################


class GCNEncoder(nn.Module):
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, dropout):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2_mu = GraphConvolution(n_hid, n_latent)
        self.gc2_sig = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout
        # try adding batch normalization?

    def forward(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        log_sig = self.gc2_sig(x, adj)
        return mu, torch.exp(log_sig)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7


    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj


class InnerProductDecoderMasked(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, adj_mat, beta=0.5):
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

    def __init__(self, data, n_hidden, n_latent, dropout, masked=False, beta=1e-2, subsampling=False):
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
        self.n_subsample = 2 * self.n_edges
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)
        self.subsampling = subsampling

        # Layers 
        self.dropout = dropout
        self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
        if masked:
            self.decoder = InnerProductDecoderMasked(self.dropout, self.adj_norm, beta)
        else:
            self.decoder = InnerProductDecoder(self.dropout)

        if self.subsampling:
            print("Using subsampling instead of reweighting")
            self.sample = get_subsampler(self.adj_labels)


    def model(self):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # Setup hyperparameters for prior p(z)
        z_mu    = ng_zeros([self.n_samples, self.n_latent])
        z_sigma = ng_ones([self.n_samples, self.n_latent])
        # sample from prior
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)
        # decode the latent code z
        z_adj = self.decoder(z)

        # Subsampling
        if self.subsampling:
            with pyro.iarange("data", self.n_subsample, subsample=self.sample()) as ind:
                pyro.observe('obs', dist.bernoulli, self.adj_labels.view(1, -1)[0][ind], z_adj.view(1, -1)[0][ind])
        # Reweighting
        else:
            with pyro.iarange("data"):
                pyro.observe('obs', weighted_bernoulli, self.adj_labels.view(1, -1), z_adj.view(1, -1), weight=self.pos_weight)

    def guide(self):
        # register PyTorch model 'encoder' w/ pyro
        pyro.module("encoder", self.encoder)
        # Use the encoder to get the parameters use to define q(z|x)
        z_mu, z_sigma = self.encoder(self.x, self.adj_norm)
        # Sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)


    def get_embeddings(self):
        z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
        # Put encoder back into training mode 
        self.encoder.train()
        return z_mu



########################################################################################


def get_vgae_embeddings(args, adj, features, threshold=0.6, 
                        laplace_reg=False, verbose=False, t=None):
    """
    args - dotdict
    adj - numpy array (or sparse csr)
    features - numpy array
    threshold - for accuracy evaluation on val and test
    """

    pyro.clear_param_store()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    N, D = features.shape
    print('{} nodes in graph, {} features per node'.format(N, D))

    adj_orig = adj

    if t is None:
        t = preprocess_train_test_split(adj, args.seed)
    features = Variable(make_sparse(features))

    
    data = {
        'adj_norm'  : t['adj_train_norm'],
        'adj_labels': t['adj_train_labels'],
        'features'  : features,
    }

    gae = GAE(data,
              n_hidden=args.n_hidden,
              n_latent=args.n_latent,
              dropout=args.dropout,
              masked=args.masked,
              subsampling=args.subsampling)

    optimizer = Adam({"lr": args.lr, "betas": (0.95, 0.999)})

    svi = SVI(gae.model, gae.guide, optimizer, loss="ELBO")

    ## ## ## ## ## ##  Additional loss ## ## ## ## ## ## ##
    
    ## only the encoder is learnable
    optimizer_e = torch.optim.Adam(gae.encoder.parameters(), lr=1e-5)

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    # Results
    results = defaultdict(list)

    print('Training starts')
    # Full batch training loop
    epoch_iter = range(args.num_epochs) if verbose else log_progress(list(range(args.num_epochs)))
    for epoch in epoch_iter:
        epoch_loss = 0. # initialize loss accumulator
        epoch_loss += svi.step()  # do ELBO gradient and accumulate loss


        ## ## ## ## ## ##  Additional loss ## ## ## ## ## ## ##
        if laplace_reg:
            optimizer_e.zero_grad()
            emb = gae.get_embeddings()
            ## LAPLACIAN REGULARIZATION / DUAL / CLUSTERING / ETC
            mat = SparseMM.apply(gae.L_norm, emb)
            loss_enc = 2 * torch.trace(torch.transpose(emb, 0, 1) @ mat)
            loss_enc.backward()
            optimizer_e.step()

        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


        # report training diagnostics
        if args.subsampling:
            normalized_loss = epoch_loss / float(2 * n_edges)
        else:
            normalized_loss = epoch_loss / (2 * N * N)
        
        results['train_elbo'].append(normalized_loss)

        # Training loss
        emb = gae.get_embeddings()
        accuracy, roc_curr, ap_curr, f1_curr, logloss = eval_gae_lp(t['val_edges'], t['val_edges_false'],
                                                           emb, adj_orig, threshold=threshold)
        
        results['accuracy_train'].append(accuracy)
        results['roc_train'].append(roc_curr)
        results['ap_train'].append(ap_curr)
        results['f1_train'].append(f1_curr)

        if verbose:
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(normalized_loss),
                  "train_acc=", "{:.5f}".format(accuracy), "val_roc=", "{:.5f}".format(roc_curr),
                  "val_ap=", "{:.5f}".format(ap_curr), "val_f1=", "{:.5f}".format(f1_curr))

        # Test loss
        if epoch % args.test_freq == 0:
            emb = gae.get_embeddings()
            accuracy, roc_score, ap_score, f1score, logloss = eval_gae_lp(t['test_edges'],
                                                                          t['test_edges_false'],
                                                                          emb, adj_orig)
            results['accuracy_test'].append(accuracy)
            results['roc_test'].append(roc_score)
            results['ap_test'].append(ap_score)
            results['f1_test'].append(f1score)

    print("Optimization Finished!")
    
    # Test loss
    emb = gae.get_embeddings()
    accuracy, roc_score, ap_score, f1score, logloss = eval_gae_lp(t['test_edges'], t['test_edges_false'],
                                                         emb, adj_orig,threshold=threshold, verbose=verbose)
    print('Test Accuracy: {:.5f}'.format(accuracy))
    print('Test ROC score: {:.5f}'.format(roc_score))
    print('Test AP score: {:.5f}'.format(ap_score))
    print('Test F1 score: {:.5f}'.format(f1score))
    print('Test logloss: {:.5f}'.format(logloss))

    # Plot
    if verbose:
        plot_results(results, args.test_freq, path=args.dataset_str + "_results.png")


    info = {'train_edges': t['train_edges']}

    return emb, info

