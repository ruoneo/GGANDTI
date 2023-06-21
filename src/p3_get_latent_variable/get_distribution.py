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


for dataset in config.datasets:
    config.dataset_temp = dataset  # This is a shared temporary variable that controls whether the reconstruction loss of X is considered
    for i in range(1):
        print("Currently working on datasets: " + dataset)
        # 读数据
        adj_orig = utils.read_orig(dataset)
        adj, _, _, _ = utils.read_dataset(dataset, i)  # , val_edges, val_edges_false
        features = utils.read_features(dataset)
        if config.features == 0:
            features = sp.identity(features.shape[0])
        if dataset == 'luo':
            config.learning_rate = 0.001
        features = utils.sparse_to_tuple(features.tocoo())

        # Standardize the training set
        adj_norm = utils.normalization(adj)

        # Define placeholders
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'adj_x': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        # Create Encoder
        model = GCNModelVAE(placeholders, features[2][1], adj.shape[0], features[1].shape[0])

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            opt = OptimizerVAE(preds=[model.reconstructions,model.reconstructions_x],
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                               adj_x=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_x'], validate_indices=False), [-1]),
                               model=model, num_nodes=adj.shape[0],
                               pos_weight=pos_weight,
                               norm=norm)
        # Initializing a session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_label = adj + utils.sp.eye(adj.shape[0])  # The adjacency matrix plus the identity matrix to embed self-information.
        adj_label_A = utils.sparse_to_tuple(adj_label)
        adj_label_x = features

        # Encoding
        for epoch in tqdm(range(config.epochs)):
            t = time.time()
            # Construct feed dictionary
            feed_dict = utils.construct_feed_dict(adj_norm, adj_label_A,adj_label_x, features, placeholders)
            feed_dict.update({placeholders['dropout']: config.dropout})

            # Run the single weight update
            outs = sess.run([model.z, opt.opt_op, opt.grads_vars], feed_dict=feed_dict)

            if epoch == config.epochs - 1:  # the last time
                utils.write_distribution_to_file(outs[0], epoch, i, dataset)

        print("Encoding Finished!")

        tf.reset_default_graph()
        sess.close()
        gc.collect()

        print("The latent variables of the dataset" + dataset + "have been obtained!\n".format(i))
