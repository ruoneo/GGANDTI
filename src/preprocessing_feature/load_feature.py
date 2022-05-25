import numpy as np
import scipy.sparse as sp


def load_yam_feature(dataset):
    dp = np.loadtxt('../../data/RawData/Yamanishi/{}_admat_dgc.txt'.format(dataset), dtype=str, delimiter='\t')[1:, 1:].astype(np.float).T
    dd = np.loadtxt('../../data/RawData/Yamanishi/{}_simmat_dc.txt'.format(dataset), dtype=str, delimiter='\t')[1:, 1:].astype(np.float)
    pp = np.loadtxt('../../data/RawData/Yamanishi/{}_simmat_dg.txt'.format(dataset), dtype=str, delimiter='\t')[1:, 1:].astype(np.float)
    feature = np.vstack((np.hstack((dd, dp)), np.hstack((dp.T, pp))))
    return sp.lil_matrix(feature)


def load_luo_feature(dataset):
    dp = np.loadtxt('../../data/RawData/luo/mat_drug_protein.txt'.format(dataset), dtype=float)
    dd = np.loadtxt('../../data/RawData/luo/Similarity_Matrix_Drugs.txt'.format(dataset), dtype=float)
    pp = np.loadtxt('../../data/RawData/luo/Similarity_Matrix_Proteins.txt'.format(dataset), dtype=float) / 100
    feature = np.vstack((np.hstack((dd, dp)), np.hstack((dp.T, pp))))
    return sp.lil_matrix(feature)
