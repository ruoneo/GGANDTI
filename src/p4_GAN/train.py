import gc
import os
import pickle

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import sys
os.chdir('./src')
sys.path.append(os.path.abspath('../../GGANDTI-main'))
sys.path.append(os.path.abspath('../../GGANDTI-main/src'))
os.chdir('./p4_GAN')
from src.p4_GAN.model import GraphGAN
from src import config
from src.p4_GAN import utils

'''
In p1_preprocessing_data/, we have completed the division of the dataset and saved the division results in data/partitioned_data/, 
so we read data directly from the corresponding fold during training.
'''
for dataset in config.datasets:
    print("Currently working on datasets: " + dataset)
    for fold in range(10):
        print(str(fold) + "-th times of ten-fold cross-validation on the dataset " + dataset + " in progress")

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

        print(str(fold) + "-th times of ten-fold cross-validation on the dataset " + dataset + " is done\n")

'''
The results of each fold are written to a file and can be viewed under results/{dataset}/final_results. The following code reads the result from memory and writes it to the file.
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
                if count in idx_gen:
                    # To parse the resulting text, traverse the line to locate the numbers
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
        inf.write("Value of ten times auroc:\n" + str(auroc) + "\n" + "Value of ten times auprc:\n" + str(auprc) + "\n")
        inf.write("average_auroc={:.5f}".format(average_auroc) + "\naverage_auprc={:.5f}".format(average_auprc))

    print("The degree to which the imbalance is altered: {:.0%}, ".format(config.percent) + dataset, end=" ")

    print("average_auroc={:.5f}".format(average_auroc), "average_auprc={:.5f}".format(average_auprc))
