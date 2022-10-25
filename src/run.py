import os
import sys
import time

from src import config

'''
This is a script that performs a full ten-fold CVs training and validation process.

This script needs to be executed ten times in order to get ten-fold cross-validation results (results are recorded in results/{dataset}/final_results_{n}.txt). 
Corresponding parameters need to be configured in src/config.py.

This script is optional. Another way to execute training and verification: The execution process of training and verification can be executed sequentially from the second-level directory under src/.
'''
sys.path.append(os.path.abspath('p1_preprocessing_data'))
sys.path.append(os.path.abspath('p3_get_latent_variable'))
sys.path.append(os.path.abspath('p4_GAN'))
sys.path.append(os.path.abspath('p5_other'))

print("改变不平衡的比例：{:.0%}, 第{}次十折交叉验证开始..".format(config.percent, config.seed))

t_start = time.time()

# 设置基准路径,记录初始导入模块
os.chdir(os.path.abspath('.'))

# 处理数据
os.chdir('p1_preprocessing_data')
print("处理数据...")
from p1_preprocessing_data import process_data

print("处理数据完成!\n")

# 处理特征
os.chdir('../p2_preprocessing_feature')
print("处理特征...")
from p2_preprocessing_feature import process_feature

print("处理特征完成!\n")

# 学习分布
os.chdir('../p3_get_latent_variable')
print("编码特征...")
from p3_get_latent_variable import get_distribution

print("特征编码完成!\n")

# GAN训练
os.chdir("../p4_GAN")
if not os.path.exists(config.model_log):
    os.makedirs(config.model_log)
print("GraphGAN训练...")
from p4_GAN import train

print("GraphGAN训练完成!")
print("改变不平衡的比例：{:.0%}, 第{}次十折交叉验证结束!".format(config.percent, config.seed))

print("用时：", time.time() - t_start)
