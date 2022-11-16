import os
import pickle
import sys
os.chdir('./src')
sys.path.append(os.path.abspath('../../GGANDTI-main'))
sys.path.append(os.path.abspath('../../GGANDTI-main/src'))
os.chdir('./p2_preprocessing_feature')
from src import config
from src.p2_preprocessing_feature.load_feature import load_yam_feature, load_luo_feature

for dataset in config.datasets:
    # feature: lil_matrix
    if dataset == 'luo':
        feature = load_luo_feature(dataset)
    else:
        feature = load_yam_feature(dataset)

    # 保存特征
    path = "../../data/partitioned_data/{}/feature/".format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(feature, open(path + dataset + "_feature.pkl", 'wb'))

    print("ok")
