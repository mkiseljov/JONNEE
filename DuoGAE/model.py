# PyTorch model

from DuoGAE.diff2vec import Diff2Vec
from DuoGAE.node2vec import Node2Vec
from DuoGAE.vgae import GAE, eval_gae_lp, GAE_NP
from DuoGAE.layers import SparseMM
from DuoGAE.utils import log_progress, sparse_mx_to_torch_sparse_tensor
from DuoGAE.preprocessing import make_incidence_matrix, \
                                 preprocess_train_test_split

from DuoGAE.visualization import plot_results
from datetime import datetime
import numpy as np
import networkx as nx
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

from collections import defaultdict
from tqdm import tqdm, tqdm_notebook

wrapper = tqdm if __name__ == "__main__" else tqdm_notebook

from DuoGAE.datasets import to_dual




class DuoGAE(object):
    def __init__(self):
        """
        TODO
        """
        # instantiate two GAEs
        pass

    """
        At the moment, functions like fit_and_score
        and works on full data
    """
    def eval_lp(self, dataset, emb_type="node2vec",
                feat_dim=16, weighted_seq=True,
                n_hidden=32, n_latent=16,
                dropout=0.1, dropout_dual=0.1,
                mask=1e-2, mask_dual=1e-2,
                lambd=0.2, lambd_dual=0.4, co_reg=0.1,
                num_epochs=100, test_freq=10,
                lr=1e-2, lr_reg=1e-5,
                alternate_training=False,
                train_test_split=(0.85, 0.05, 0.1),
                seed=0, threshold=0.6, verbose=False):
        """
        ######## Main link prediction learning method ########
        TODO: Something's up with the seed

        ## Input data ##
        :param: dataset - object of descendant class if GraphDataset
        
        ## Feature learning ##
        :param: emb_type="node2vec" - sequence-based model
                (one of "node2vec", "diff2vec", "dummy")
                if dataset has features, dummy means using them
        :param: feat_dim=16 - dimensions from learned sequence-based model
        :param: weighted_seq=True - whether to use weighted feature generator
        
        ## Masking ##
        # Set to zeros to disable
        :param: mask=1e-2
        :param: mask_dual=1e-2

        ## Laplace ##
        # Set to zeros to disable each component
        :param: lambd=2 - setting to zero means no Laplace reg for GAE
        :param: lambd_dual=4 - setting to zero means no Laplace reg for GAE^*
        :param: co_reg=1 - setting to zero disables dual training
        
        ## GAE fitting ##
        :param: n_hidden=32 - hidden layer of GCN
        :param: n_latent=16 - dimensions of resulting representation (d)
        :param: dropout=0.1 - regularization of primal GAE
        :param: dropout_dual=0.1 - regularization of dual GAE
        :param: num_epochs=100 - total training epochs
        :param: lr=1e-2 - learning rate for both GAEs
        :param: lr_reg=1e-5 - learning for the extra loss
        :param: alternate_training=False - if True, on even iters updates primal,
                                    on odd dual
        :param: seed=0 - seed for train-test split etc
        :param: threshold=0.6 - prediction threshold
        :param: verbose=False

        ## Rubbish parameters ##
        :param: test_freq=10

        """

        # train two GAE instances with common loss functions
        
        pyro.clear_param_store()
        np.random.seed(seed)
        torch.manual_seed(seed)     

        dataset.preprocess_train_test_split(seed,
                                train_test_split)

        if emb_type == "dummy":
            data = self.make_dummy_features(dataset)
        # elif emb_type == "node2vec":
        #     dataset.make_weighted()
        #     data = self.make_features(dataset, emb_type, 
        #                               dimensions=feat_dim,
        #                               weighted=weighted_seq)
        else:
            dataset.make_weighted()
            data = self.make_features(dataset, emb_type,
                                      dimensions=feat_dim,
                                      weighted=weighted_seq)


        features = data['features']
        adj = data['adj']
        N, D = features.shape

        features = Variable(torch.FloatTensor(features))
        commat = make_incidence_matrix(dataset.node2ind_map, dataset.edgelist)
        degrees = commat.to_dense().sum(dim=1)  # degree of each vertex

        data = {
            'adj_norm'  : dataset.t['adj_train_norm'],
            'adj_labels': dataset.t['adj_train_labels'],
            'features'  : features,
        }

        ###################################################
        dataset_dual = to_dual(dataset.t['adj_train'])

        dataset_dual.preprocess_train_test_split(seed,
                                     train_test_split)

        # if emb_type == "node2vec":
        # dataset_dual.make_weighted()

        if emb_type == "dummy":
            data_dual = self.make_dummy_features(dataset_dual)
        else:
            dataset_dual.make_weighted()
            data_dual = self.make_features(dataset_dual, emb_type,
                                           dimensions=feat_dim,
                                           weighted=weighted_seq)

        features_dual = data_dual['features']
        adj_dual = data_dual['adj']
        N_dual, D_dual = features_dual.shape
        
        features_dual = Variable(torch.FloatTensor(features_dual))
        commat_dual = make_incidence_matrix(dataset_dual.node2ind_map,
                                            dataset_dual.edgelist)
        degrees_dual = commat_dual.to_dense().sum(dim=1)  # degree of each vertex

        data_dual = {
            'adj_norm'  : dataset_dual.t['adj_train_norm'],
            'adj_labels': dataset_dual.t['adj_train_labels'],
            'features'  : features_dual,
        }

        ########### Primal model and optimizer ###########

        gae = GAE(data,
                  n_hidden=n_hidden,
                  n_latent=n_latent,
                  dropout=dropout,
                  mask=mask)
        self.gae = gae
        
        optimizer = Adam({"lr": lr, "betas": (0.95, 0.999)})
        svi = SVI(gae.model, gae.guide, optimizer, loss="ELBO")
        ## as only the encoder is learnable
        optimizer_e = torch.optim.Adam(gae.encoder.parameters(), lr=lr_reg)

        ################ Same but for dual ###############

        gae_dual = GAE(data_dual,
                       n_hidden=n_hidden,
                       n_latent=n_latent,
                       dropout=dropout_dual,
                       mask=mask_dual)
        self.dual_gae = gae_dual
        optimizer_dual = Adam({"lr": lr, "betas": (0.95, 0.999)})
        svi_dual = SVI(gae_dual.model, gae_dual.guide, optimizer_dual, loss="ELBO")
        # TODO: select this learning rate
        optimizer_e_dual = torch.optim.Adam(gae_dual.encoder.parameters(), lr=lr_reg)

        ##################################################

        commat_train_edges_var = Variable(make_incidence_matrix(len(dataset.node2ind_map),
                                                                dataset.t['train_edges']))


        ##################################################
        #                 Training loop                  #
        ##################################################

        # Results
        results = defaultdict(list)

        # Full batch training loop
        try:
            epoch_iter = wrapper(range(num_epochs), leave=False)
        except:
            epoch_iter = tqdm(range(num_epochs))
        for epoch in epoch_iter:
            t = datetime.now()
            epoch_loss = 0.           # initialize loss accumulator
            epoch_loss += svi.step()  # do ELBO gradient and accumulate loss

            epoch_loss_dual = 0.
            epoch_loss_dual += svi_dual.step()

            ## ## ## ## ## ##  Additional loss ## ## ## ## ## ## ##
            # TODO: add lambda and lambda_dual regularization coefficients

            optimizer_e.zero_grad()
            optimizer_e_dual.zero_grad()

            emb = gae.get_embeddings()
            emb_dual = gae_dual.get_embeddings()

            ## LAPLACIAN REGULARIZATION (later: DUAL / CLUSTERING / ETC)
            mat = SparseMM.apply(gae.L_norm, emb)
            loss_enc = torch.trace(torch.transpose(emb, 0, 1) @ mat)
            # loss_enc.backward()

            ## LAPLACIAN REGULARIZATION FOR DUAL
            mat_dual = SparseMM.apply(gae_dual.L_norm, emb_dual)
            loss_enc_dual = torch.trace(torch.transpose(emb_dual, 0, 1) @ mat_dual)
            # loss_enc_dual.backward()

            ## Common loss for primal and dual
            #            n x E_tr x E_tr x d
            # averages = (commat @ emb_dual) / degrees[:, np.newaxis]
            averages = SparseMM.apply(commat_train_edges_var, emb_dual) \
                                            / Variable(degrees[:, np.newaxis])
            loss_common = ((emb - averages) ** 2).sum()  # Frobenius
            # loss_common.backward()

            # variables have accumulated the gradients, perform steps along them
            if alternate_training:
                if epoch % 2 == 0:
                    loss = lambd * loss_enc \
                            + co_reg * loss_common
                else:
                    loss = lambd_dual * loss_enc_dual \
                            + co_reg * loss_common
            else:
                loss = lambd * loss_enc \
                       + lambd_dual * loss_enc_dual \
                       + co_reg * loss_common

            if epoch == 0:
                calibrator = 1.0
                if loss.data.numpy() > 1e-10:
                    calibrator = epoch_loss / loss.data.numpy()
                # print(type(calibrator))
                # assert isinstance(calibrator, float)
            else:
                loss *= calibrator
                # print(loss.data.numpy()[0], epoch_loss)

            loss *= lambd
            loss.backward()

            optimizer_e.step()
            optimizer_e_dual.step()

            ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

            # report training diagnostics
            normalized_loss = epoch_loss / (2 * N * N)
            loss_np = loss.data.numpy() / (2 * N * N)

            results['train_elbo'].append(normalized_loss)
            results['train_added'].append(loss_np)
            results["time_per_epoch"].append((datetime.now() - t).total_seconds())

            # Loss on validation
            results_curr = self.evaluate(dataset, 'val', threshold, verbose)
            results['accuracy_train'].append(results_curr['accuracy'])
            results['roc_train'].append(results_curr['roc_score'])
            results['ap_train'].append(results_curr['ap_score'])
            results['f1_train'].append(results_curr['f1_score'])

            if verbose:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_ELBO=", "{:.5f}".format(normalized_loss),
                      "train_np=", "{:.5f}".format(loss_np),
                      "val_acc=", "{:.5f}".format(results_curr['accuracy']),
                      "val_roc=", "{:.5f}".format(results_curr['roc_score']),
                      "val_ap=", "{:.5f}".format(results_curr['ap_score']),
                      "val_f1=", "{:.5f}".format(results_curr['f1_score']))

            # Test loss
            if epoch % test_freq == 0:
                results_curr = self.evaluate(dataset, 'test', threshold, verbose)
                results['accuracy_test'].append(results_curr['accuracy'])
                results['roc_test'].append(results_curr['roc_score'])
                results['ap_test'].append(results_curr['ap_score'])
                results['f1_test'].append(results_curr['f1_score'])


        ################ DONE #################

        plot_results(results, test_freq, path="results.png", show=verbose)
        return results


    def fit(dataset, **kwargs):
        """
        Like eval_lp, but no train_test_split
        """
        raise NotImplementedError


    def predict(self, test_data):
        """
        :param data: Dataset instance
        
        """
        raise NotImplementedError


    def evaluate(self, dataset, mode='test', threshold=0.5, verbose=0):
        """
        Evaluate trained model on testing set 
        and return metrics dictionary.
        """
        if dataset.noeval:
            test_results = {'accuracy': -1,
                            'roc_score': -1,
                            'ap_score': -1,
                            'f1_score': -1,
                            'logloss': -1}
            return test_results

        emb = self.gae.get_embeddings()
        accuracy, roc_score, ap_score, f1_score, logloss = eval_gae_lp(dataset.t[mode+'_edges'],
                                                                      dataset.t[mode+'_edges_false'],
                                                                      emb, dataset.adj,
                                                                      threshold=threshold,
                                                                      verbose=verbose)
        test_results = {'accuracy': accuracy, 
                        'roc_score': roc_score,
                        'ap_score': ap_score,
                        'f1_score': f1_score,
                        'logloss': logloss}
        
        if verbose:
            for key, value in test_results.items():
                print(mode + " " + key, value)
        
        return test_results


    def make_dummy_features(self, dataset):
        data = dataset.get_data()
        if data['features'] is None:
            data['features'] = np.identity(data['adj'].shape[0])
        return data


    def make_features(self, dataset, emb_type="node2vec",
                      dimensions=16, weighted=True):
        # if already has features, concatenate, 
        # otherwise just replace them
        if emb_type == "node2vec":
            mod = Node2Vec(dimensions=dimensions, weighted=weighted)
        elif emb_type == "diff2vec":
            mod = Diff2Vec(dimensions=dimensions, weighted=weighted)
        else:
            raise Exception("Unknown seq-based embedding %s passed." % emb_type)

        # mod.learn_embeddings(dataset)
        nx_G_train = nx.from_scipy_sparse_matrix(dataset.t["adj_train"])
        assert nx_G_train.number_of_nodes() == dataset.n
        mod.learn_embeddings(nx_G_train)
        emb = mod.get_embeddings()

        # replace:
        data = dataset.get_data()
        if data['features'] is None:
            data['features'] = emb
        else:
            # TODO: concatenate? or use only old feats?
            #       (using only new ones looses info)
            # print(emb.shape)
            # print(data['features'].shape)
            data['features'] = np.concatenate((emb, data['features']), axis=1)
        
        # data['features'] = (emb - emb.mean(axis=1)[:, np.newaxis]) \
        #                         / (1e-5 + emb.std(axis=1)[:, np.newaxis])
        assert data['features'].shape[0] == data['adj'].shape[0]

        return data

















###########################################################################





















class DuoGAE_NP(object):
    def __init__(self):
        """
        TODO
        """
        # instantiate two GAEs
        pass

    """
        At the moment, functions like fit_and_score
        and works on full data
    """
    def eval_lp(self, dataset, emb_type="node2vec",
                feat_dim=16, weighted_seq=True,
                n_hidden=32, n_latent=16,
                dropout=0.1, dropout_dual=0.1,
                mask=1e-2, mask_dual=1e-2,
                lambd=2, lambd_dual=4, co_reg=1,
                num_epochs=100, test_freq=10,
                lr=1e-2, lr_reg=1e-5,
                alternate_training=False,
                seed=0, threshold=0.6, verbose=False):
        """
        ######## Main link prediction learning method ########
        TODO: Something's up with the seed

        ## Input data ##
        :param: dataset - object of descendant class if GraphDataset
        
        ## Feature learning ##
        :param: emb_type="node2vec" - sequence-based model
                (one of "node2vec", "diff2vec", "dummy")
                if dataset has features, dummy means using them
        :param: feat_dim=16 - dimensions from learned sequence-based model
        :param: weighted_seq=True - whether to use weighted feature generator
        
        ## Masking ##
        # Set to zeros to disable
        :param: mask=1e-2
        :param: mask_dual=1e-2

        ## Laplace ##
        # Set to zeros to disable each component
        :param: lambd=2 - setting to zero means no Laplace reg for GAE
        :param: lambd_dual=4 - setting to zero means no Laplace reg for GAE^*
        :param: co_reg=1 - setting to zero disables dual training
        
        ## GAE fitting ##
        :param: n_hidden=32 - hidden layer of GCN
        :param: n_latent=16 - dimensions of resulting representation (d)
        :param: dropout=0.1 - regularization of primal GAE
        :param: dropout_dual=0.1 - regularization of dual GAE
        :param: num_epochs=100 - total training epochs
        :param: lr=1e-2 - learning rate for both GAEs
        :param: lr_reg=1e-5 - learning for the extra loss
        :param: alternate_training=False - if True, on even iters updates primal,
                                    on odd dual
        :param: seed=0 - seed for train-test split etc
        :param: threshold=0.6 - prediction threshold
        :param: verbose=False

        ## Rubbish parameters ##
        :param: test_freq=10

        """

        # train two GAE instances with common loss functions
        
        pyro.clear_param_store()
        np.random.seed(seed)
        torch.manual_seed(seed)     

        dataset.preprocess_train_test_split(seed)

        if emb_type == "dummy":
            data = self.make_dummy_features(dataset)
        # elif emb_type == "node2vec":
        #     dataset.make_weighted()
        #     data = self.make_features(dataset, emb_type, 
        #                               dimensions=feat_dim,
        #                               weighted=weighted_seq)
        else:
            dataset.make_weighted()
            data = self.make_features(dataset, emb_type,
                                      dimensions=feat_dim,
                                      weighted=weighted_seq)


        features = data['features']
        adj = data['adj']
        N, D = features.shape

        features = Variable(torch.FloatTensor(features))
        commat = make_incidence_matrix(dataset.node2ind_map, dataset.edgelist)
        degrees = commat.to_dense().sum(dim=1)  # degree of each vertex

        data = {
            'adj_norm'  : dataset.t['adj_train_norm'],
            'adj_labels': dataset.t['adj_train_labels'],
            'features'  : features,
        }

        ###################################################
        dataset_dual = to_dual(dataset.t['adj_train'])

        dataset_dual.preprocess_train_test_split(seed)

        # if emb_type == "node2vec":
        # dataset_dual.make_weighted()

        if emb_type == "dummy":
            data_dual = self.make_dummy_features(dataset_dual)
        else:
            dataset_dual.make_weighted()
            data_dual = self.make_features(dataset_dual, emb_type,
                                           dimensions=feat_dim,
                                           weighted=weighted_seq)

        features_dual = data_dual['features']
        adj_dual = data_dual['adj']
        N_dual, D_dual = features_dual.shape
        
        features_dual = Variable(torch.FloatTensor(features_dual))
        commat_dual = make_incidence_matrix(dataset_dual.node2ind_map,
                                            dataset_dual.edgelist)
        degrees_dual = commat_dual.to_dense().sum(dim=1)  # degree of each vertex

        data_dual = {
            'adj_norm'  : dataset_dual.t['adj_train_norm'],
            'adj_labels': dataset_dual.t['adj_train_labels'],
            'features'  : features_dual,
        }

        ########### Primal model and optimizer ###########

        gae = GAE_NP(data,
                  n_hidden=n_hidden,
                  n_latent=n_latent,
                  dropout=dropout)

        self.gae = gae
        
        adj_dense = adj.todense()
        optimizer = torch.optim.Adam(gae.encoder.parameters(), lr=lr)
        mask_var = Variable(torch.FloatTensor(adj_dense * mask + 1))

        ################ Same but for dual ###############

        gae_dual = GAE_NP(data_dual,
                       n_hidden=n_hidden,
                       n_latent=n_latent,
                       dropout=dropout_dual)

        adj_dense_dual = adj_dual.todense()
        optimizer_dual = torch.optim.Adam(gae_dual.encoder.parameters(), lr=lr)
        mask_var_dual = Variable(torch.FloatTensor(adj_dense_dual * mask_dual + 1))

        ##################################################

        commat_train_edges_var = Variable(make_incidence_matrix(len(dataset.node2ind_map),
                                                                dataset.t['train_edges']))


        ##################################################
        #                 Training loop                  #
        ##################################################

        # Results
        results = defaultdict(list)

        Adj = Variable(sparse_mx_to_torch_sparse_tensor(adj))
        Adj_dual = Variable(sparse_mx_to_torch_sparse_tensor(adj_dual))
        Adj_dense = Variable(torch.FloatTensor(adj_dense))
        Adj_dense_dual = Variable(torch.FloatTensor(adj_dense_dual))

        # print(Adj.data.shape, Adj_dual.data.shape, mask_var.data.shape, mask_var_dual.data.shape)

        # Full batch training loop
        epoch_iter = wrapper(range(num_epochs), leave=False)
        for epoch in epoch_iter:
            t = datetime.now()
            loss_mse = ((gae(features, Adj) - Adj_dense) ** 2 * mask_var).sum()
            loss_mse_dual = ((gae_dual(features_dual, Adj_dual) \
                            - Adj_dense_dual) ** 2\
                             * mask_var_dual).sum()
            optimizer.zero_grad()
            optimizer_dual.zero_grad()

            ## ## ## ## ## ##  Additional loss ## ## ## ## ## ## ##

            emb = gae.get_embeddings()
            emb_dual = gae_dual.get_embeddings()

            ## LAPLACIAN REGULARIZATION (later: DUAL / CLUSTERING / ETC)
            mat = SparseMM.apply(gae.L_norm, emb)
            loss_enc = torch.trace(torch.transpose(emb, 0, 1) @ mat)

            ## LAPLACIAN REGULARIZATION FOR DUAL
            mat_dual = SparseMM.apply(gae_dual.L_norm, emb_dual)
            loss_enc_dual = torch.trace(torch.transpose(emb_dual, 0, 1) @ mat_dual)

            ## Common loss for primal and dual
            #            n x E_tr x E_tr x d
            averages = SparseMM.apply(commat_train_edges_var, emb_dual) \
                                    / Variable(degrees[:, np.newaxis])
            loss_common = ((emb - averages) ** 2).sum()  # Frobenius
            if epoch == 0:
                calibrator = 1.0
                val = (lambd * loss_enc \
                        + lambd_dual * loss_enc_dual \
                        + co_reg * loss_common).data.numpy()[0]

                if val > 1e-7:
                    calibrator = (loss_mse + loss_mse_dual).data.numpy()[0] \
                                    / val
                    calibrator = float(calibrator)
                # print(calibrator, type(calibrator))
                # assert isinstance(calibrator, float)

            if alternate_training:
                if epoch % 2 == 0:
                    loss = loss_mse \
                            + calibrator * ( \
                            + lambd * loss_enc \
                            + co_reg * loss_common \
                            )
                else:
                    loss = loss_mse_dual \
                            + calibrator * ( \
                           + lambd_dual * loss_enc_dual \
                            + co_reg * loss_common \
                            )
            else:
                loss = loss_mse + loss_mse_dual \
                       + calibrator * ( \
                           + lambd * loss_enc \
                           + lambd_dual * loss_enc_dual \
                           + co_reg * loss_common \
                        )

            loss.backward()

            optimizer.step()
            optimizer_dual.step()

            ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

            # report training diagnostics
            normalized_loss = (loss_mse + loss_mse_dual).data.numpy()[0] / (2 * N * N)
            loss_np = loss.data.numpy() / (2 * N * N)

            results['train_elbo'].append(normalized_loss)
            results['train_added'].append(loss_np)
            results["time_per_epoch"].append((datetime.now() - t).total_seconds())

            # Loss on validation
            results_curr = self.evaluate(dataset, 'val', threshold, verbose)
            results['accuracy_train'].append(results_curr['accuracy'])
            results['roc_train'].append(results_curr['roc_score'])
            results['ap_train'].append(results_curr['ap_score'])
            results['f1_train'].append(results_curr['f1_score'])

            if verbose:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_ELBO=", "{:.5f}".format(normalized_loss),
                      "train_np=", "{:.5f}".format(loss_np),
                      "val_acc=", "{:.5f}".format(results_curr['accuracy']),
                      "val_roc=", "{:.5f}".format(results_curr['roc_score']),
                      "val_ap=", "{:.5f}".format(results_curr['ap_score']),
                      "val_f1=", "{:.5f}".format(results_curr['f1_score']))

            # Test loss
            if epoch % test_freq == 0:
                results_curr = self.evaluate(dataset, 'test', threshold, verbose)
                results['accuracy_test'].append(results_curr['accuracy'])
                results['roc_test'].append(results_curr['roc_score'])
                results['ap_test'].append(results_curr['ap_score'])
                results['f1_test'].append(results_curr['f1_score'])


        ################ DONE #################
        
        plot_results(results, test_freq, path="results.png")
        return results


    def fit(dataset, **kwargs):
        """
        Like eval_lp, but no train_test_split
        """
        raise NotImplementedError


    def predict(self, test_data):
        """
        :param data: Dataset instance
        
        """
        raise NotImplementedError


    def evaluate(self, dataset, mode='test', threshold=0.5, verbose=0):
        """
        Evaluate trained model on testing set 
        and return metrics dictionary.
        """
        emb = self.gae.get_embeddings()
        accuracy, roc_score, ap_score, f1_score, logloss = eval_gae_lp(dataset.t[mode+'_edges'],
                                                                      dataset.t[mode+'_edges_false'],
                                                                      emb, dataset.adj,
                                                                      threshold=threshold,
                                                                      verbose=verbose)
        test_results = {'accuracy': accuracy, 
                        'roc_score': roc_score,
                        'ap_score': ap_score,
                        'f1_score': f1_score,
                        'logloss': logloss}
        
        if verbose:
            for key, value in test_results.items():
                print(mode + " " + key, value)
        
        return test_results


    def make_dummy_features(self, dataset):
        data = dataset.get_data()
        if data['features'] is None:
            data['features'] = np.identity(data['adj'].shape[0])
        return data


    def make_features(self, dataset, emb_type="node2vec",
                      dimensions=16, weighted=True):
        # if already has features, concatenate, 
        # otherwise just replace them
        if emb_type == "node2vec":
            mod = Node2Vec(dimensions=dimensions, weighted=weighted)
        elif emb_type == "diff2vec":
            mod = Diff2Vec(dimensions=dimensions, weighted=weighted)
        else:
            raise Exception("Unknown seq-based embedding %s passed." % emb_type)

        # mod.learn_embeddings(dataset)
        nx_G_train = nx.from_scipy_sparse_matrix(dataset.t["adj_train"])
        assert nx_G_train.number_of_nodes() == dataset.n
        mod.learn_embeddings(nx_G_train)
        emb = mod.get_embeddings()

        # replace:
        data = dataset.get_data()
        if data['features'] is None:
            data['features'] = emb
        else:
            # TODO: concatenate? or use only old feats?
            #       (using only new ones looses info)
            data['features'] = np.concatenate((emb, data['features']), axis=1)
            # TODO: Standardize?
        
        # data['features'] = (emb - emb.mean(axis=1)[:, np.newaxis]) \
        #                         / (1e-5 + emb.std(axis=1)[:, np.newaxis])
        assert data['features'].shape[0] == data['adj'].shape[0]

        return data


