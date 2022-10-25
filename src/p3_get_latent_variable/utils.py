import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import auc

import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def read_dataset(dataset, i):
    adj_train = pickle.load(open("../../data/partitioned_data/{0}/{1}fold/{0}_adj_train.pkl".format(dataset, i), "rb"))
    train_edges = np.loadtxt("../../data/partitioned_data/{0}/{1}fold/{0}_vgae_train.txt".format(dataset, i), dtype=int)
    test_edges = np.loadtxt("../../data/partitioned_data/{0}/{1}fold/{0}_vgae_test.txt".format(dataset, i), dtype=int)
    test_edges_false = np.loadtxt("../../data/partitioned_data/{0}/{1}fold/{0}_vgae_test_neg.txt".format(dataset, i), dtype=int)
    return adj_train, train_edges, test_edges, test_edges_false  # , val_edges, val_edges_false


def read_features(dataset):
    pickle_file = open("../../data/partitioned_data/{0}/feature/{0}_feature.pkl".format(dataset), "rb")
    features = pickle.load(pickle_file)
    pickle_file.close()
    return features


def read_orig(dataset):
    pickle_file = open("../../data/partitioned_data/{0}/orig/{0}_adj_orig.pkl".format(dataset), "rb")
    adj_orig = pickle.load(pickle_file)
    pickle_file.close()
    return adj_orig


def normalization(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def write_distribution_to_file(distributions, epoch, i, dataset):
    for i in range(10):
        indexs = list(range(distributions.shape[0]))
        filename = '../../data/partitioned_data/{0}/{1}fold/{0}'.format(dataset, i)
        nodes = get_nodes(filename + '_train.txt', filename + '_test.txt')
        with open(filename + '_pre_train.emb', 'w') as f:
            f.write(str(len(nodes)) + " " + str(distributions.shape[1]) + "\n")  # distributions.shape[0]
            for distribution, index in zip(distributions, indexs):
                if index in nodes:
                    f.write(str(index) + " " + " ".join([str(j) for j in distribution]) + "\n")
    return None


def get_nodes(train_name, test_name):
    train = np.loadtxt(train_name, dtype=int)
    nodes = set(train.flatten())
    return nodes


def get_auc(a, b):
    return abs(auc(a, b))
