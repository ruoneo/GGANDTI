import os
import pickle

import numpy as np

from src import config
import scipy.sparse as sp

from load_data import load_yam_data, change_unbalanced, load_luo_data
from utils import divide_vgae_datasets, sparse_to_tuple, divide_datasets

for dataset in config.datasets:
    g = os.walk(r"../../data/partitioned_data/{}".format(dataset))
    for path, dir_list, file_list in g:
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))
    print("清除 {} 缓存完成!".format(dataset))

    # Load data 得到一个邻接矩阵,双向边
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

    # 获得不同不平衡性的数据
    adj = change_unbalanced(adj, config.percent, dp_line, dataset)

    # Store original adjacency matrix (without diagonal entries) for later  保存原始邻接矩阵(不含对角线项)以备后用
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)  # 假设对角线有元素，去除对角线
    adj_orig.eliminate_zeros()  # 假设有0，移除矩阵中的0
    path = "../../data/partitioned_data/{}/orig/".format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(adj_orig, open(path + dataset + "_adj_orig.pkl", 'wb'))
    np.savetxt(path + dataset + "_adj_orig.txt", adj_orig.A, fmt='%d')

    # 为获取嵌入划分数据, 划分数据集, 并记录边
    for i in range(10):
        # Remove diagonal elements                                      # 删除对角线元素
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # 梅开二度
        adj.eliminate_zeros()
        # Check that diag is zero:                                      # 检查diag是否为零：
        assert np.diag(adj.todense()).sum() == 0

        # 为graphgan划分数据
        g_adj = adj[0:dp_line, dp_line:]
        g_edges = sparse_to_tuple(g_adj)[0]
        g_num_test = int(np.floor(g_edges.shape[0] / 10.))  # np.floor()是向下取整。测试集10分之一，训练集20分之一
        g_num_val = int(np.floor(g_edges.shape[0] / 20.))

        adj_pd, train_edges, test_edges, test_edges_false = divide_datasets(g_adj, g_edges, g_num_test, i, dp_line)
        adj[0:dp_line, dp_line:] = adj_pd

        # 将训练集分给vgae
        edges = sparse_to_tuple(sp.triu(adj))[0]
        edges_all = sparse_to_tuple(adj)[0]  # 将邻接矩阵转换成三元组，然后只取坐标，即所有的边
        num_test = int(np.floor(edges.shape[0] / 10.))  # np.floor()是向下取整。测试集10分之一，训练集20分之一
        num_val = int(np.floor(edges.shape[0] / 20.))

        adj_train, vgae_train_edges, vgae_test_edges, vgae_test_edges_false, val_edges, val_edges_false = divide_vgae_datasets(adj, edges, edges_all, num_test, num_val, i)
        # 保存划分好的数据
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
