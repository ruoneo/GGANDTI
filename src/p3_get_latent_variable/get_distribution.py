import gc
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_eager_execution()

from encoder import GCNModelVAE, OptimizerVAE
from tqdm import tqdm
import scipy.sparse as sp

from src.p3_get_latent_variable import utils
from src import config

# Encoder Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

for dataset in config.datasets:
    for i in range(1):
        print("当前正在操作的数据集: " + dataset)
        # 读数据
        adj_orig = utils.read_orig(dataset)
        adj, _, _, _ = utils.read_dataset(dataset, i)  # , val_edges, val_edges_false
        features = utils.read_features(dataset)
        if config.features == 0:
            features = sp.identity(features.shape[0])
        features = utils.sparse_to_tuple(features.tocoo())

        # 对训练集做标准化
        adj_norm = utils.normalization(adj)

        # 定义placeholders占位符
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        # Create Encoder
        model = GCNModelVAE(placeholders, features[2][1], adj.shape[0], features[1].shape[0])

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                               model=model, num_nodes=adj.shape[0],
                               pos_weight=pos_weight,
                               norm=norm)
        # 初始化会话
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_label = adj + utils.sp.eye(adj.shape[0])  # 邻接矩阵加上单位矩阵
        adj_label = utils.sparse_to_tuple(adj_label)

        # Train model
        for epoch in tqdm(range(FLAGS.epochs)):
            t = time.time()
            # Construct feed dictionary
            feed_dict = utils.construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # 运行单权重更新
            outs = sess.run([model.z, opt.opt_op, opt.grads_vars], feed_dict=feed_dict)

            if epoch == FLAGS.epochs - 1:  # 最后一次
                utils.write_distribution_to_file(outs[0], epoch, i, dataset)

        print("Encoding Finished!")

        tf.reset_default_graph()
        sess.close()
        gc.collect()

        print(dataset + "数据集的潜在变量已获得!\n".format(i))
