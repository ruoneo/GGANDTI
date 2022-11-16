import os
import pickle

import numpy as np

import sys
os.chdir('./src')
sys.path.append(os.path.abspath('../../GGANDTI-main'))
sys.path.append(os.path.abspath('../../GGANDTI-main/src'))
os.chdir('./p1_preprocessing_data')
from src import config
import scipy.sparse as sp

from load_data import load_yam_data, change_unbalanced, load_luo_data
from utils import divide_vgae_datasets, sparse_to_tuple, divide_datasets

for dataset in config.datasets:
    g = os.walk(r"../../data/partitioned_data/{}".format(dataset))
    for path, dir_list, file_list in g:
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))
    print("Clearing the {}'s cache is complete!".format(dataset))

    # Load data. You get an adjacency matrix, bidirectional edges
    if dataset == 'luo':
        adj, dp_line = load_luo_data(dataset)
    else:
        adj, dp_line = load_yam_data(dataset)

    if not os.path.exists("../../data/partitioned_data"):
        os.mkdir("../../data/partitioned_data")
    if not os.path.exists("../../data/partitioned_data/{}".format(dataset)):
        os.mkdir("../../data/partitioned_data/{}".format(dataset))
    if not os.path.exists("../../data/partitioned_data/{}/orig".format(dataset)):
        os.mkdir("../../data/partitioned_data/{}/orig/".format(dataset))
    np.savetxt("../../data/partitioned_data/{}/orig/dp_line.txt".format(dataset), np.array([dataset, str(dp_line)]), fmt='%s')

    # Obtain data of different unevenness
    adj = change_unbalanced(adj, config.percent, dp_line, dataset)

    # Store original adjacency matrix (without diagonal entries) for later.  Save the original adjacency matrix (without diagonal entries) for later use.
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)  # Let's say the diagonal has elements, let's get rid of the diagonal
    adj_orig.eliminate_zeros()  # Let's say I have 0, remove 0 from the matrix.
    path = "../../data/partitioned_data/{}/orig/".format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(adj_orig, open(path + dataset + "_adj_orig.pkl", 'wb'))
    np.savetxt(path + dataset + "_adj_orig.txt", adj_orig.A, fmt='%d')

    # To obtain the embedded partition data, partition the data set, and record the edge.
    for i in range(10):
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # second bloom
        adj.eliminate_zeros()
        # Check that diag is zero.
        assert np.diag(adj.todense()).sum() == 0

        # Divide the data for graphgan
        g_adj = adj[0:dp_line, dp_line:]
        g_edges = sparse_to_tuple(g_adj)[0]
        g_num_test = int(np.floor(g_edges.shape[0] / 10.))  # np.floor() is rounded down. Test set 1/10, training set 1/20
        g_num_val = int(np.floor(g_edges.shape[0] / 20.))

        adj_pd, train_edges, test_edges, test_edges_false = divide_datasets(g_adj, g_edges, g_num_test, i, dp_line)
        adj[0:dp_line, dp_line:] = adj_pd

        # Assign the training set to vgae
        edges = sparse_to_tuple(sp.triu(adj))[0]
        edges_all = sparse_to_tuple(adj)[0]  # Convert the adjacency matrix to a triple, and then just take the coordinates, that is, all the edges.
        num_test = int(np.floor(edges.shape[0] / 10.))  # np.floor() is rounded down. Test set 1/10, training set 1/20.
        num_val = int(np.floor(edges.shape[0] / 20.))

        adj_train, vgae_train_edges, vgae_test_edges, vgae_test_edges_false, val_edges, val_edges_false = divide_vgae_datasets(adj, edges, edges_all, num_test, num_val, i)
        # Save the partitioned data
        path = "../../data/partitioned_data/{}/{}fold/".format(dataset, i)
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(adj_train, open(path + dataset + "_adj_train.pkl", 'wb'))

        np.savetxt(path + dataset + "_vgae_train.txt", vgae_train_edges, fmt='%d')
        np.savetxt(path + dataset + "_vgae_val.txt", val_edges, fmt='%d')
        np.savetxt(path + dataset + "_vgae_val_neg.txt", val_edges_false, fmt='%d')
        np.savetxt(path + dataset + "_vgae_test.txt", vgae_test_edges, fmt='%d')
        np.savetxt(path + dataset + "_vgae_test_neg.txt", vgae_test_edges_false, fmt='%d')

        np.savetxt(path + dataset + "_train.txt", vgae_train_edges, fmt='%d')
        np.savetxt(path + dataset + "_pd_train.txt", train_edges, fmt='%d')
        np.savetxt(path + dataset + "_val.txt", val_edges, fmt='%d')
        np.savetxt(path + dataset + "_val_neg.txt", val_edges_false, fmt='%d')
        np.savetxt(path + dataset + "_test.txt", test_edges, fmt='%d')
        np.savetxt(path + dataset + "_test_neg.txt", test_edges_false, fmt='%d')

    print("OK")
