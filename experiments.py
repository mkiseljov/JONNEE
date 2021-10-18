# DuoGAE imports and settings

from DuoGAE.model import DuoGAE, DuoGAE_NP
from DuoGAE.datasets import *
from DuoGAE.visualization import plot_points_tsne
from DuoGAE.diff2vec import Diff2Vec
from DuoGAE.node2vec import Node2Vec
from DuoGAE.vgae import eval_gae_lp

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE

import torch
from torch.autograd import Variable

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,\
                             RandomForestClassifier

from tqdm import tqdm

# other imports
from collections import defaultdict
from itertools import product
from copy import copy
import json
import warnings
warnings.filterwarnings("ignore")

import pickle as pkl


def interval_evaluation_lp(dataset, model, params, seeds):
    """
    Average across different runs on a fixed dataset.
    :param: dataset - GraphDataset instance
    :param: model - DuoGAE object
    :param: params - dictionary of DuoGAE.eval_lp() keyword args

    :return: res={"accuracy": [0.7, 0.65, ...], ...} (defaultdict)
    """
    res = defaultdict(list)

    for seed in seeds:
        params['seed'] = seed
        _ = model.eval_lp(dataset, **params)
        res_run = model.evaluate(dataset)
        for met in res_run:
            res[met].append(res_run[met])
    return res


def to_mean_and_std(results):
    res = {}
    for r in results:
        case = results[r]
        tmp = {}
        for met in case:
            tmp[met] = {'mean': np.mean(case[met]), 
                        'std':  np.std(case[met])}
        res[r] = copy(tmp)
    return res

# ! mkdir results

RES_FOLDER="./results/"

def load_result(name):
    with open(RES_FOLDER+name+".json") as fp:
        js = json.load(fp=fp)
    return js


def save_result(results, name):
    with open(RES_FOLDER+name+".json", "w") as fp:
        json.dump(results, fp=fp)




###############################################################
######################### TEST SUITES #########################
###############################################################


def duogae_vs_all_test(run_all):
    # run us on all available datasets, using optimal parameters

    # dataset_gens = {'hepth': HepThDataset, 'astro': AstroPhDataset,
    #                 'hse': HSEDataset, 'ppi': PPIDataset,
    #                 # 'facebook': FacebookDataset,
    #                 'blog_catalog': BlogCatalogDataset}

    subgraph_configs = [{'size': 2000, 'seed': 30},
                        {'size': 2000, 'seed': 40},
                        {'size': 2000, 'seed': 50}]

    subgraph_configs_small = [{'size': 1000, 'seed': 30},
                              {'size': 1000, 'seed': 40},
                              {'size': 1000, 'seed': 50}]
    # subgraph_configs_small = [{'size': 1000, 'seed': 30}]

    tt_seed = 42

    ds = [2, 3, 4, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7]
    RESULTS_F1 = defaultdict(list)
    RESULTS_ROC = defaultdict(list)
    dataset_gens = {'ppi': PPIDataset}
    #########################################################
    # DuoGAE

    # for d in ds:
        # print("DuoGAE,", d)
    seeds = [tt_seed]
    d = 16

    # dataset_gens = {'blog_catalog': BlogCatalogDataset}
    # dataset_gens = {'astro': AstroPhDataset}
    # 'facebook': FacebookDataset,

    tuned_params = {'emb_type': 'diff2vec',
                    'num_epochs': 400}

    tuned_params['n_latent'] = d
    tuned_params['n_hidden'] = 2 * d
    tuned_params['feat_dim'] = d
    for name in dataset_gens:
        print(name)
        model = DuoGAE()
        result = defaultdict(list)
        if name in ['facebook', 'blog_catalog']:
            confs = subgraph_configs_small
        else:
            confs = subgraph_configs
        for conf in tqdm(confs):
            # conf['compress_to'] = d
            dataset = dataset_gens[name](**conf)
            res = interval_evaluation_lp(dataset, model,
                                         params=tuned_params,
                                         seeds=seeds)
            for met in res:
                result[met].extend(res[met])

        # RESULTS_F1['duogae'].append(np.mean(result["f1_score"]))
        # RESULTS_ROC['duogae'].append(np.mean(result["roc_score"]))
        # done, saving
        save_result({met: {"mean": np.mean(v), "std": np.std(v)}
                    for met, v in result.items()}, "duogae_diff_"+name)

    ############################################################
    def learn(model_gen, model_name):
        ### Learn others: we need only the embeddings
        for name in dataset_gens:
            print(name)
            res = defaultdict(list)
            if name in ['facebook', 'blog_catalog']:
                confs = subgraph_configs_small
            else:
                confs = subgraph_configs
            for i, sg in enumerate(confs):
                dataset = dataset_gens[name](**sg)
                dataset.preprocess_train_test_split(seed=tt_seed)
                mod = model_gen()
                nx_G_train = nx.from_scipy_sparse_matrix(dataset.t["adj_train"])
                assert nx_G_train.number_of_nodes() == dataset.n
                mod.learn_embeddings(nx_G_train)
                emb = Variable(torch.FloatTensor(mod.get_embeddings()))
                accuracy, roc_score, ap_score, f1score, logloss = eval_gae_lp(dataset.t['test_edges'],
                                                                              dataset.t['test_edges_false'],
                                                                              emb, dataset.adj,
                                                                              threshold=0.5, verbose=False)
                res['accuracy_score'].append(accuracy)
                res['roc_score'].append(roc_score)
                res['ap_score'].append(ap_score)
                res['f1_score'].append(f1score)
                res['logloss'].append(logloss)

            RESULTS_F1[model_name].append(np.mean(res["f1_score"]))
            RESULTS_ROC[model_name].append(np.mean(res["roc_score"]))
            # save_result({met: {"mean": np.mean(v), "std": np.std(v)}
            #             for met, v in res.items()}, model_name + "_" + name)

    if run_all:
        ## GAE: just set everything to zero
        # seeds = [tt_seed]

        # reset_params = {'emb_type': 'dummy',
        #                 'mask': 0.0, 'mask_dual': 0.0,
        #                 'lambd': 0.0, 'lambd_dual': 0.0,
        #                 'co_reg': 0.0}

        # for d in ds:
        #     print("GAE,", d)
        #     reset_params['n_latent'] = d
        #     reset_params['n_hidden'] = 2 * d
        #     reset_params['feat_dim'] = d
        #     for name in dataset_gens:
        #         print(name)
        #         model = DuoGAE()
        #         result = defaultdict(list)
        #         if name in ['facebook', 'blog_catalog']:
        #             confs = subgraph_configs_small
        #         else:
        #             confs = subgraph_configs
        #         for conf in confs:
        #             dataset = dataset_gens[name](**conf)
        #             res = interval_evaluation_lp(dataset, model,
        #                                          params=reset_params,
        #                                          seeds=seeds)
        #             for met in res:
        #                 result[met].extend(res[met])

        #         RESULTS_F1['gae'].append(np.mean(result["f1_score"]))
        #         RESULTS_ROC['gae'].append(np.mean(result["roc_score"]))
                # done, saving
                # save_result({met: {"mean": np.mean(v), "std": np.std(v)}
                #             for met, v in result.items()}, "gae_"+name)

        #########################################################
        # Other guys

        for d in ds:
            print("Other models for dim:", d)
            learn(lambda: Diff2Vec(dimensions=d), "diff2vec_plain")
            learn(lambda: Node2Vec(dimensions=d), "node2vec_plain")
            learn(lambda: HOPE(d=d, beta=0.01), "HOPE")
            learn(lambda: GraphFactorization(d=d, max_iter=10000,
                                             eta=1*10**-4, regu=1.0), "GrF")

    # pkl.dump(RESULTS_F1, open('RESULTS_F1_others.pkl', 'wb'))
    # pkl.dump(RESULTS_ROC, open('RESULTS_ROC_others.pkl', 'wb'))
    # RESULTS = pkl.load(open('RESULTS.pkl', 'rb')) - loading them back
    print("Done test duogae_vs_all")
    #########################################################




###############################################################
###################           RUN         #####################
###############################################################

if __name__ == "__main__":
    duogae_vs_all_test(run_all=0)












