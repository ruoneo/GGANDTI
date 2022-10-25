import numpy as np
import scipy.sparse as sp

from src import config


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def divide_vgae_datasets(adj, edges, edges_all, num_test, num_val, i):
    # 构建具有10%正向链接的测试集的函数
    # 注：拆分是随机的，结果可能与论文中报告的数字略有偏差。

    if i == 9:
        start_test = num_test * i
        end_test = edges.shape[0]
        start_val = 0
        end_val = num_val
    else:
        start_test = num_test * i
        end_test = num_test * (i + 1)
        start_val = end_test
        end_val = end_test + num_val

    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(config.seed)
    np.random.shuffle(edges)
    val_edge_idx = all_edge_idx[start_val:end_val]
    test_edge_idx = all_edge_idx[start_test:end_test]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx]), axis=0)  # , val_edge_idx

    def ismember(a: list, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])  # 随机生成横坐标
        idx_j = np.random.randint(0, adj.shape[0])  # 随机生成纵坐标
        if idx_i == idx_j:  # 对角线的不要
            continue
        if ismember([idx_i, idx_j], edges_all):  # 是已知边不要
            continue
        if test_edges_false:  # 已选负边不要，a-b或b-a有一个是都不要
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:  # 对角线不要
            continue
        if ismember([idx_i, idx_j], edges_all):  # 是已知边不要
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(test_edges, train_edges)

    # Re-build adj matrix   重建邻接矩阵
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T  # 因为train_edges是单向的,所以把它变成对称的

    # NOTE: these edge lists only contain single direction of edge!  注意：这些边列表只包含边的单一方向！
    return adj, train_edges, test_edges, np.array(test_edges_false),val_edges, np.array(val_edges_false)


def divide_datasets(adj, edges, num_test, i, dp_line):
    if i == 9:
        start_test = num_test * i
        end_test = edges.shape[0]
    else:
        start_test = num_test * i
        end_test = num_test * (i + 1)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(config.seed)
    np.random.shuffle(edges)
    test_edge_idx = all_edge_idx[start_test:end_test]
    test_edges = edges[test_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx]), axis=0)  # , val_edge_idx

    def ismember(a: list, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])  # 随机生成横坐标
        idx_j = np.random.randint(0, adj.shape[1])  # 随机生成纵坐标
        if idx_i == idx_j:  # 自身不要
            continue
        if ismember([idx_i, idx_j], edges):  # 是已知边不要
            continue
        test_edges_false.append([idx_i, idx_j])

    adj_pd = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    # 把列索引编号加上dp_line
    def add_index(edges):
        edges = np.array(edges)
        colu = edges[:, 1] + dp_line
        edges[:, 1] = colu
        return edges

    train_edges = add_index(train_edges)
    test_edges = add_index(test_edges)
    test_edges_false = add_index(test_edges_false)

    return adj_pd, train_edges, test_edges, test_edges_false
