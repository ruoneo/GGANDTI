import gc
import os
import pickle

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from src.p4_GAN.model import GraphGAN
from src import config
from src.p4_GAN import utils

'''
In p1_preprocessing_data/, we have completed the division of the dataset and saved the division results in data/partitioned_data/, 
so we read data directly from the corresponding fold during training.
'''
for dataset in config.datasets:
    print("正在使用数据集: " + dataset + " 进行训练...")
    for fold in range(10):
        print("正在进行数据集" + dataset + "的第" + str(fold) + "折交叉验证")

        utils.deal_filename(dataset, fold)

        f = open("../../data/partitioned_data/{0}/{1}fold/{0}_adj_train.pkl".format(dataset, fold), 'rb')
        config.shape = pickle.load(f).shape
        f.close()

        config.dp_line = np.loadtxt('../../data/partitioned_data/{}/orig/dp_line.txt'.format(dataset), dtype=str)[1].astype(int)

        config.lr_gen = config.datasets_best_paras[dataset][0]  # learning rate for the generator
        config.lr_dis = config.datasets_best_paras[dataset][1]  # learning rate for the discriminator
        config.n_sample_gen = config.datasets_best_paras[dataset][2]  # number of samples for the generator
        config.n_epochs_gen = config.datasets_best_paras[dataset][3]  # number of inner loops for the generator
        config.n_epochs_dis = config.datasets_best_paras[dataset][4]  # number of inner loops for the discriminator

        config.gen_interval = config.n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
        config.dis_interval = config.n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations

        '''
        训练及验证的核心过程
        
        The core process of training and validation
        '''
        graph_gan = GraphGAN()
        graph_gan.train()
        tf.reset_default_graph()
        graph_gan.sess.close()
        del graph_gan
        gc.collect()

        rocs_key = sorted(config.aurocs, reverse=True)
        prcs_key = sorted(config.auprcs, reverse=True)

        np.savetxt('../../results/{}/{}fold/fpr.txt'.format(dataset, fold), config.aurocs[rocs_key[0]][0])
        np.savetxt('../../results/{}/{}fold/tpr.txt'.format(dataset, fold), config.aurocs[rocs_key[0]][1])
        np.savetxt('../../results/{}/{}fold/recall.txt'.format(dataset, fold), config.auprcs[prcs_key[0]][0])
        np.savetxt('../../results/{}/{}fold/precision.txt'.format(dataset, fold), config.auprcs[prcs_key[0]][1])

        print("数据集" + dataset + "的第" + str(fold) + "折交叉验证完成!\n")

'''
每一折的结果都被写入了文件, 可以在 results/{dataset}/final_results 下查看

The results of each fold are written to a file and can be viewed under results/{dataset}/final_results
'''
for dataset in config.datasets:
    auroc = []
    auprc = []
    for i in range(10):
        filename = "../../results/" + dataset + '/' + str(i) + 'fold/' + dataset + ".txt"

        best_auroc = 0
        best_auprc = 0
        idx_gen = list(range(0, 62, 2))
        idx_dis = list(range(1, 62, 2))

        with open(filename, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if count in idx_gen:  # 说明遍历的是gen
                    # 遍历line定位数字
                    position = utils.search_position(line)
                    auroc_value = float(line[position['auroc_start']:position['auroc_end']])
                    auprc_value = float(line[position['auprc_start']:])
                    if auroc_value > best_auroc:
                        best_auroc = auroc_value
                    if auprc_value > best_auprc:
                        best_auprc = auprc_value
                count += 1
        auroc.append(best_auroc)
        auprc.append(best_auprc)

    average_auroc = sum(auroc) / len(auroc)
    average_auprc = sum(auprc) / len(auprc)

    path = "../../results/{}/final_results/".format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "final_result_{}.txt".format(config.seed), "w") as inf:
        inf.write("十次auroc值:\n" + str(auroc) + "\n" + "十次auprc值:\n" + str(auprc) + "\n")
        inf.write("average_auroc={:.5f}".format(average_auroc) + "\naverage_auprc={:.5f}".format(average_auprc))

    print("改变不平衡的比例: {:.0%}, ".format(config.percent) + dataset, end=" ")

    print("average_auroc={:.5f}".format(average_auroc), "average_auprc={:.5f}".format(average_auprc))
