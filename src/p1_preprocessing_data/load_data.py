import os
import random

import numpy as np
import scipy.sparse as sp


def load_luo_data(dataset):
    dp = np.loadtxt('../../data/RawData/luo/mat_drug_protein.txt'.format(dataset), dtype=int)
    dd = np.loadtxt('../../data/RawData/luo/mat_drug_drug.txt'.format(dataset), dtype=int)
    pp = np.loadtxt('../../data/RawData/luo/mat_protein_protein.txt'.format(dataset), dtype=int)
    adj = np.vstack((np.hstack((dd, dp)), np.hstack((dp.T, pp))))
    return sp.csr_matrix(adj + sp.eye(adj.shape[0])), dd.shape[0]


def load_yam_data(dataset):
    dp = np.loadtxt('../../data/RawData/Yamanishi/{}_admat_dgc.txt'.format(dataset), dtype=str, delimiter='\t')[1:, 1:].astype(np.int).T
    dd = np.loadtxt('../../data/RawData/Yamanishi/{}_simmat_dc.txt'.format(dataset), dtype=str, delimiter='\t')[1:, 1:].astype(np.float)
    pp = np.loadtxt('../../data/RawData/Yamanishi/{}_simmat_dg.txt'.format(dataset), dtype=str, delimiter='\t')[1:, 1:].astype(np.float)

    z_score_dd = (dd - np.mean(dd)) / np.std(dd)
    dd = np.where(z_score_dd >= 1.64, 1, 0)
    z_score_pp = (pp - np.mean(pp)) / np.std(pp)
    pp = np.where(z_score_pp >= 1.64, 1, 0)

    adj = np.vstack((np.hstack((dd, dp)), np.hstack((dp.T, pp))))
    return sp.csr_matrix(adj), dd.shape[0]


def is_symmetry(adj):
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != adj[j][i]:
                return False
    return True


def is_1_diag(adj):
    if sum(np.diagonal(adj)) != adj.shape[0]:
        return False
    return True


def change_unbalanced(adj, percent, dp_line, dataset):
    """
    note: percent控制屏蔽掉的节点所占的百分比
    :param adj:
    :param percent:
    :return: 返回去除部分已知关联的邻接矩阵
    """
    # 判断是否对称
    # assert is_symmetry(adj.A)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape) + sp.eye(adj.shape[0])
    # 判断对角线是否全为1
    assert is_1_diag(adj.A)
    adj = (sp.triu(adj) + sp.triu(adj).T - sp.eye(adj.shape[0])).A

    row = list(range(0, dp_line))
    col = list(range(dp_line, adj.shape[0]))

    idx = []
    for i in row:
        for j in col:
            if i != j and adj[i][j] == 1:
                idx.append((i, j))
    num = int(np.floor(percent * len(idx)))
    count = 0
    # random.seed(config.seed)
    while count < num:
        row, col = random.choice(idx)
        idx.remove((row, col))
        adj[row][col] = 0
        adj[col][row] = 0
        count += 1

    # idx = []
    # for i in range(adj.shape[0]):
    #     for j in range(i + 1, adj.shape[0]):
    #         if adj[i][j] == 1:
    #             idx.append((i, j))
    # num = int(np.floor(percent * len(idx)))
    # count = 0
    # # random.seed(config.seed)
    # while count < num:
    #     row, col = random.choice(idx)
    #     idx.remove((row, col))
    #     adj[row][col] = 0
    #     adj[col][row] = 0
    #     count += 1

    # 保存改变不平衡性后新的dp
    new_dp = adj[0:dp_line, dp_line:]
    # if not os.path.exists('../../data/partitioned_data/{0}/feature'.format(dataset)):
    #     os.mkdir('../../data/partitioned_data/{0}/feature'.format(dataset))
    # np.savetxt('../../data/partitioned_data/{0}/feature/{0}_new_admat_dgc.txt'.format(dataset), new_dp, fmt='%d', delimiter='\t')
    return sp.csr_matrix(adj.astype(np.int))
