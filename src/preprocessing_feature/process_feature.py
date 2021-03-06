import os
import pickle

from src import config
from src.preprocessing_feature.load_feature import load_yam_feature, load_luo_feature

for dataset in config.datasets:
    # feature: lil_matrix
    if dataset == 'luo':
        feature = load_luo_feature(dataset)
    else:
        feature = load_yam_feature(dataset)

    # 保存特征
    path = "../../data/datasets/{}/feature/".format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(feature, open(path + dataset + "_feature.pkl", 'wb'))

    print("ok")
