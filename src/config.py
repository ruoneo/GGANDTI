"""
All hyperparameters
"""

datasets = ['e']  # , 'e','ic', 'gpcr', 'nr', 'luo'
percent = 0  # Contribution parameters of imbalance. example: 0 indicates the original data without changing the imbalance degree; 0.5 represents the data with 50% of the positive sample masked
seed = 0  # The seeds 0-9 used in dividing the dataset represent the (0-9)-th times 10-fold cross validation. The number here refers to the seed of the dataset that shuffles each CV (necessary) and also refers to the number in the file name under results/{e}/final_results (unnecessary).

"""
Some parameters of training part. 
"""
modes = ["gen", "dis"]
lambda_dis = 1e-5
lr_dis = 0.05    # learning rate for the discriminator
lambda_gen = 1e-5
lr_gen = 1e-3    # learning rate for the generator
n_sample_gen = 5
n_emb = 64
n_epochs = 30
n_epochs_dis = 20
n_epochs_gen = 20
dis_interval = 20
gen_interval = 20
batch_size_dis = 128
batch_size_gen = 128
lambda_con = 1e-3
constraint = True
# attention
attention = True
lr_att = 0.001
attention_size = 8
w_decay = 0.00001

window_size = 2
update_ratio = 1

'''
default parameters, Read in sequence in the order described later. learning rate for the generator, learning rate for the discriminator,number of samples for the generator, number of inner loops for the generator,number of inner loops for the discriminator
'''
datasets_best_paras = {
    'e': [1e-3, 0.05, 5, 20, 20],
    'ic': [0.001, 0.1, 5, 20, 15],
    'gpcr': [0.01, 0.01, 5, 15, 20],
    'nr': [0.05, 0.01, 15, 15, 20],
    'luo': [1e-3, 0.1, 5, 20, 20]
}

'''
The following parameters with 'None' are assigned in src/p4_GAN/utils.py. If you don't need to change the directory hierarchy, do not change the following parameters..
'''
train_filename = None
test_filename = None
test_neg_filename = None
val_filename = None
val_neg_filename = None
pretrain_emb_filename_d = None
pretrain_emb_filename_g = None
initial_features = None
result_filename = None
emb_filenames = None
cache_filename = None
model_log = "../../saved_model/"

# save
save_model = False
load_model = False  # whether to load existed model
save_steps = 10

# other
count = 0
shape = None
dp_line = None

recalls = {}
aurocs = {}
auprcs = {}

"""
parameters of encoder 
"""
learning_rate = 0.01
epochs = 200
hidden1 = 128
hidden2 = 64
weight_decay = 0
dropout = 0
features = 1
