# Plot results

import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_results(results, test_freq, path='results.png', show=False):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    x_axis_train_added = range(len(results['train_added']))
    # Elbo and added loss
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'], label="ELBO")
    ax.plot(x_axis_train_added, results['train_added'], label="L1+L2+L3")
    ax.set_ylabel('Loss on train')
    ax.set_title('Loss')
    ax.legend(loc='upper right')

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
    if show:
        plt.show()
    else:
        fig.savefig(path)


def plot_points_tsne(emb, seed=42, labels=None, figsize=(12, 8)):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40,
    			n_iter=600, random_state=seed)
    
    tsne_results = tsne.fit_transform(emb)

    vis_x = tsne_results[:, 0]
    vis_y = tsne_results[:, 1]

    plt.figure(figsize=figsize)
    plt.scatter(vis_x, vis_y)
    plt.show()



def plot_points(emb, seed=42, labels=None, figsize=(12, 8)):
    vis_x = emb[:, 0]
    vis_y = emb[:, 1]

    plt.figure(figsize=figsize)
    plt.scatter(vis_x, vis_y)
    plt.show()



def plot_with_classes(emb, edges, labels, figsize=(14, 10)):
    # dots
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=600)
    tsne_results = tsne.fit_transform(emb)

    # plotting
    vis_x = tsne_results[:, 0]
    vis_y = tsne_results[:, 1]
    y_data = labels

    plt.figure(figsize=figsize)
    plt.scatter(vis_x, vis_y, c=y_data,
        cmap=plt.cm.get_cmap("jet", 39))
    # plt.colorbar(ticks=range(39))

    for edge in edges:
        a, b = edge
        a, b = int(a), int(b)
        plt.plot([vis_x[a], vis_x[b]], [vis_y[a], vis_y[b]],
                 color='gray', alpha=0.1)

    plt.show()

