import numpy as np
from sklearn.metrics import recall_score, roc_curve, auc, precision_recall_curve

from src.p4_GAN import utils
from src import config


class LinkPredictEval(object):
    def __init__(self, embed_filename, test_filename, test_neg_filename, n_node, n_embed):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename  # each line: node_id1, node_id2
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = utils.read_embeddings(embed_filename, n_node=n_node, n_embed=n_embed)

    def eval_link_prediction(self):
        test_edges = utils.read_edges_from_file(self.test_filename)
        test_edges_neg = utils.read_edges_from_file(self.test_neg_filename)
        test_edges.extend(test_edges_neg)

        config.count = config.count + 1     # 用来定位

        # may exists isolated point
        score_res = []
        for i in range(len(test_edges)):
            score_res.append(np.dot(self.emd[test_edges[i][0]], self.emd[test_edges[i][1]]))
        test_label = np.array(score_res)
        median = np.median(test_label)
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0  # test_label代表预测标签
        true_label = np.zeros(test_label.shape)
        true_label[0: len(true_label) // 2] = 1  # true_label代表真实标签

        # 计算召回率
        recall = recall_score(true_label, test_label)
        # config.recalls[recall] = [true_label, test_label]

        score_res = utils.softmax(np.array(score_res))

        # 计算AUC
        fpr, tpr, thresholds = roc_curve(true_label, score_res)
        auroc = auc(fpr, tpr)
        config.aurocs[auroc] = [fpr, tpr, config.count]

        # 计算AUPR
        precision_cur, recall_cur, thresholds = precision_recall_curve(true_label, score_res)
        auprc = auc(recall_cur, precision_cur)
        config.auprcs[auprc] = [recall_cur, precision_cur, config.count]

        return auroc, auprc
