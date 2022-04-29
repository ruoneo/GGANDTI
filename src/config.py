"""
预处理部分的参数
"""
datasets = ['luo','e','ic', 'gpcr', 'nr']  # , 'luo']  # 'luo',
percent = 0  # 不平衡性的贡献参数
seed = 9  # 划分数据集时使用的种子  0-9分别代表第0-9次十折交叉验证

"""
获取潜在变量部分的参数
"""
learning_rate = 0.01
epochs = 200
hidden1 = 128
hidden2 = 64
weight_decay = 0
dropout = 0
features = 1

"""
GraphGAN部分的参数, None表示数据待定
"""
# 训练部分的一些参数
modes = ["gen", "dis"]
lambda_dis = 1e-5
lr_dis = None
lambda_gen = 1e-5
lr_gen = None
n_sample_gen = None
n_emb = 64
n_epochs = 30
n_epochs_dis = None
n_epochs_gen = None
dis_interval = None
gen_interval = None
batch_size_dis = 128
batch_size_gen = 128
lambda_con = 1e-3
constraint = True
# 注意力机制的参数
attention = True
lr_att = 0.001
attention_hidden_size = None
attention_size = 8
attention_feature_size = None
w_decay = 0.00001
# 采样
window_size = 2
update_ratio = 1

# 一些文件位置
datasets_best_paras = {
    'e': [1e-3, 0.05, 5, 20, 20],
    'ic': [0.001, 0.1, 5, 20, 15],
    'gpcr': [0.01, 0.01, 5, 15, 20],
    'nr': [0.05, 0.01, 15, 15, 20],
    'luo': [1e-3, 0.1, 5, 20, 20]
}
train_filename = None
test_filename = None
test_neg_filename = None
pretrain_emb_filename_d = None
pretrain_emb_filename_g = None
initial_features = None
result_filename = None
emb_filenames = None
cache_filename = None
model_log = None

# 保存模型
save_model = False
load_model = False
save_steps = 10

# 一些其他的参数
count = 0
shape = None
dp_line = None

recalls = {}
aurocs = {}
auprcs = {}
