import os
import sys
import time

from src import config

sys.path.append(os.path.abspath('./preprocessing_data'))
sys.path.append(os.path.abspath('./get_latent_variable'))
sys.path.append(os.path.abspath('main'))
sys.path.append(os.path.abspath('./addition'))
for i in sys.path:
    print(i)
print("路径输出完成！\n")

print("百分比：{:.0%},第{}次十折交叉验证开始..".format(config.percent, config.seed))

t_start = time.time()

# 设置基准路径,记录初始导入模块
os.chdir(os.path.abspath('.'))

# 处理数据
os.chdir('./preprocessing_data')
print("处理数据...")
from preprocessing_data import process_data

print("处理数据完成!\n")

# 处理特征
os.chdir('../preprocessing_feature')
print("处理特征...")
from preprocessing_feature import process_feature

print("处理特征完成!\n")

# 学习分布
os.chdir('../get_latent_variable')
print("获取潜在变量...")
from get_latent_variable import get_distribution

print("获取潜在变量完成!\n")

# GAN训练
os.chdir("../GraphGAN")
print("GraphGAN训练...")
from main import train

print("GraphGAN训练完成!")
print("百分比：{:.0%},第{}次十折交叉验证结束!".format(config.percent, config.seed))

print("用时：", time.time() - t_start)
