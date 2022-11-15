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
    # Build a function with a 10% forward linked test set
    # Note: The split is random and the results may deviate slightly from the numbers reported in the paper.

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
        idx_i = np.random.randint(0, adj.shape[0])  # Randomly generate the abscissa.
        idx_j = np.random.randint(0, adj.shape[0])  # Randomly generate the y-coordinate.
        if idx_i == idx_j:  # Not the diagonal ones
            continue
        if ismember([idx_i, idx_j], edges_all):  # It's a given edge no
            continue
        if test_edges_false:  # The negative side is not chosen, either a minus b or b minus a.
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:  # No diagonal elements
            continue
        if ismember([idx_i, idx_j], edges_all):  # No known edge
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(test_edges, train_edges)

    # Re-build adj matrix
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T  # Since train_edges are one-way, make it symmetrical.

    # NOTE: these edge lists only contain single direction of edge!
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
        idx_i = np.random.randint(0, adj.shape[0])  # Randomly generate the abscissa
        idx_j = np.random.randint(0, adj.shape[1])  # Randomly generate the y-coordinate
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges):  # No known edge
            continue
        test_edges_false.append([idx_i, idx_j])

    adj_pd = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    # Add the column index number to dp_line(The boundary between drug and target features in the matrix)
    def add_index(edges):
        edges = np.array(edges)
        colu = edges[:, 1] + dp_line
        edges[:, 1] = colu
        return edges

    train_edges = add_index(train_edges)
    test_edges = add_index(test_edges)
    test_edges_false = add_index(test_edges_false)

    return adj_pd, train_edges, test_edges, test_edges_false
