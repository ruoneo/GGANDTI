import os
import sys
import time

os.chdir('./src')
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
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

print("The degree to which the imbalance is altered: {:.0%}, {}-th times of ten-fold cross-validation on the dataset is starting..".format(config.percent, config.seed))

t_start = time.time()

# Set the base path to record the initial imported module
os.chdir(os.path.abspath('../'))

# processing data
# os.chdir('p1_preprocessing_data')
print("processing data...")
from p1_preprocessing_data import process_data
print("Processing data is completed!\n")

#  Feature Processing
os.chdir('../../')
print("Feature Processing...")
from p2_preprocessing_feature import process_feature
print("Processing features complete!\n")

# Encoding adjacency matrix and initial features
os.chdir('../../')
print("Encoding adjacency matrix and initial features...")
from p3_get_latent_variable import get_distribution

#  Training GANs
os.chdir("../../")
if not os.path.exists(config.model_log):
    os.makedirs(config.model_log)
print("Training GraphGAN...")
from p4_GAN import train

print("GraphGAN Training done!\n")
print("The degree to which the imbalance is altered: {:.0%}, {}-th times of ten-fold cross-validation on the dataset is done".format(config.percent, config.seed))

print("Times：", time.time() - t_start)
