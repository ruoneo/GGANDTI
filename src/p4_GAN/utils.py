import os
import pickle

import numpy as np

from src import config


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix


def read_edges(train_filename, test_filename):
    """read data from files

    Args:
        train_filename: training file name
        test_filename: test file name

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = {}
    nodes = set()
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []

    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []

    return len(nodes), graph


def print_info(gen_auroc, gen_auprc, dis_auroc, dis_auprc, epoch=-1, dis_loss=0, gen_loss=0):
    print("Epoch:", '%04d' % (epoch + 1),
          "dis_loss={:.5f}".format(dis_loss),
          "gen_loss={:.5f}".format(gen_loss),
          "gen_auroc={:.5f}".format(gen_auroc),
          "gen_auprc={:.5f}".format(gen_auprc))


def deal_filename(dataset, fold):
    # 数据部分
    config.train_filename = "../../data/partitioned_data/{0}/{1}fold/{0}_train.txt".format(dataset, fold)
    config.test_filename = "../../data/partitioned_data/{0}/{1}fold/{0}_test.txt".format(dataset, fold)
    config.test_neg_filename = "../../data/partitioned_data/{0}/{1}fold/{0}_test_neg.txt".format(dataset, fold)
    config.pretrain_emb_filename_d = "../../data/partitioned_data/{0}/{1}fold/{0}_pre_train.emb".format(dataset, fold)
    config.pretrain_emb_filename_g = "../../data/partitioned_data/{0}/{1}fold/{0}_pre_train.emb".format(dataset, fold)
    config.initial_features = "../../data/partitioned_data/{0}/feature/{0}_feature.pkl".format(dataset)

    # result部分
    path = "../../results/{}/{}fold/".format(dataset, fold)
    if not os.path.exists(path):
        os.makedirs(path)
    config.emb_filenames = ["../../results/{0}/{1}fold/{0}_gen_.emb".format(dataset, fold),
                            "../../results/{0}/{1}fold/{0}_dis_.emb".format(dataset, fold)]
    config.result_filename = "../../results/{0}/{1}fold/{0}.txt".format(dataset, fold)
    if os.path.exists(config.result_filename):
        os.remove(config.result_filename)

    # 缓存和日志部分
    path = "../../cache/{}/{}fold/".format(dataset, fold)
    if not os.path.exists(path):
        os.makedirs(path)
    config.cache_filename = "../../cache/{0}/{1}fold/{0}.pkl".format(dataset, fold)

    # config.model_log = "../../log/"
    # if not os.path.exists(config.model_log):
    #     os.makedirs(config.model_log)


def search_position(line):
    position = {
        'auroc_start': None,
        'auroc_end': None,
        'auprc_start': None,
        'auprc_end': None
    }
    length = len(line)
    for ch, index in zip(line, range(length)):
        if ch == 'o':
            position['auroc_start'] = index + 3
        if ch == ',':
            position['auroc_end'] = index
        if ch == 'p':
            position['auprc_start'] = index + 4
    return position


def read_features(dataset):
    pickle_file = open("../../data/partitioned_data/{0}/feature/{0}_feature.pkl".format(dataset), "rb")
    features = pickle.load(pickle_file)
    pickle_file.close()
    return features