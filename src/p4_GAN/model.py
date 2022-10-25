import collections
import multiprocessing
import os
import pickle
import time

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

import tensorflow.compat.v1 as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

import tqdm
import link_prediction as lp

from src import config
from src.p4_GAN import utils


class Attention_(object):
    def __init__(self, embedding_matrix):
        self.shape = embedding_matrix.shape

        # load feature  seq = embed
        features = pickle.load(open(config.initial_features, 'rb')).A.astype(np.float32)
        self.hidden_size = self.shape[1]
        self.attention_size = config.attention_size
        self.feature_size = features.shape[1]

        with tf.variable_scope('attention'):
            self.features = tf.Variable(features)
            self.embedding_matrix = tf.placeholder(tf.float32, shape=self.shape)
            self.w1_attention = tf.Variable(tf.random_normal([self.hidden_size, self.attention_size], stddev=0.1))
            self.w2_attention = tf.Variable(tf.random_normal([self.feature_size, self.attention_size], stddev=0.1))
            self.b_attention = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            self.v_attention = tf.Variable(tf.random_normal([self.attention_size, 1], stddev=0.1))
            self.bias_mat = tf.Variable(tf.zeros([self.shape[0], self.shape[0]]))  # n个节点的偏置向量

        self.value_mid = tf.tanh(tf.add(tf.matmul(self.embedding_matrix, self.w1_attention), tf.matmul(self.features, self.w2_attention)))
        self.value_mid = tf.add(self.value_mid, self.b_attention)
        self.logits = tf.matmul(self.value_mid, self.v_attention)
        self.logits = self.logits + tf.transpose(self.logits)

        self.coefs = tf.nn.softmax(tf.nn.leaky_relu(self.logits) + self.bias_mat)
        self.z = tf.matmul(self.coefs, self.embedding_matrix)
        # self.z = tf.contrib.layers.bias_add(self.vals)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.embedding_matrix, logits=self.z)) + config.w_decay * (
                tf.nn.l2_loss(self.w1_attention) + tf.nn.l2_loss(self.w2_attention) + tf.nn.l2_loss(self.v_attention) + tf.nn.l2_loss(self.b_attention))
        self.opt = tf.train.AdamOptimizer(config.lr_att).minimize(self.loss)

        # feedback
        self.bias_vector = tf.placeholder(tf.float32, shape=[None])
        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.node_embedding = tf.nn.embedding_lookup(self.z, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.z, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias

        self.c_score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.c_score))


class _Attention(object):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

        with tf.variable_scope('attention'):
            self.W_t = tf.get_variable(name='weight',
                                       shape=self.embedding_matrix.shape,
                                       initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                       trainable=True)
        self.bias = tf.Variable(tf.zeros([self.embedding_matrix.shape[0], 1]))

        drug_id = list(range(config.dp_line))
        target_id = list(range(config.dp_line, self.embedding_matrix.shape[0]))

        self.attention_scores = tf.constant([], dtype=tf.float32)
        for i in drug_id:
            for j in target_id:
                weight1 = tf.transpose(tf.matmul(self.W_t, tf.nn.embedding_lookup(self.embedding_matrix, [j]), transpose_b=True))
                weight2 = tf.tanh(tf.add(tf.matmul(self.W_t, tf.nn.embedding_lookup(self.embedding_matrix, [i]), transpose_b=True), self.bias))
                self.attention_score = tf.matmul(weight1, weight2)
                self.attention_scores = tf.concat([self.attention_scores, tf.reshape(self.attention_score, shape=(1,))], axis=0)
                print(i, j)

        self.attention_score_softmax = tf.nn.softmax(self.attention_scores.reshape((config.dp_line, -1)))
        for i in range(self.embedding_matrix.shape[0]):
            h = tf.nn.embedding_lookup(self.embedding_matrix, i)
            self.z = tf.constant([], dtype=tf.float32)
            if i < self.attention_score_softmax.shape[0]:
                alpha_i = tf.nn.embedding_lookup(self.attention_score_softmax, i)
                for alpha in alpha_i.numpy():
                    self.z = tf.concat([self.z, alpha * h], axis=1)
            if i >= self.attention_score_softmax.shape[0]:
                alpha_j = tf.gather(self.attention_score_softmax, i - self.attention_score_softmax.shape[0], axis=1)
                for alpha in alpha_j.numpy():
                    self.z = tf.concat([self.z, alpha * h])

        self.loss = tf.losses.mean_squared_error(self.embedding_matrix, self.z)
        self.opt = tf.train.AdamOptimizer(config.lr_gen).minimize(self.loss)

    def get_attention_score(self):
        return self.z


class Discriminator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('discriminator'):  # discriminator
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))  # bias initial

        self.node_id = tf.placeholder(tf.int32, shape=[None])  #
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])  #
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)  #
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)  # node_neighbor_id行
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias

        if config.constraint:
            self.node_emd_start = tf.constant(self.node_emd_init, dtype=tf.float32)
            # constraint
            self.select = tf.random_shuffle(list(range(self.embedding_matrix.shape[0])))[0:30]
            # before training
            self.emm_start = tf.nn.embedding_lookup(self.node_emd_start, self.select)
            self.emm_start_cosine_similarity = self.compute_cosine_similarity(self.emm_start)
            # after training
            self.emm = tf.nn.embedding_lookup(self.embedding_matrix, self.select)
            self.emm_cosine_similarity = self.compute_cosine_similarity(self.emm)
            # the error
            self.difference = tf.abs(self.emm_cosine_similarity - self.emm_start_cosine_similarity)
        else:
            config.lambda_con = 0
            self.difference = tf.constant([0], dtype=tf.float32)

        if config.attention:
            self.shape = self.embedding_matrix.shape
            # load fearture
            features = pickle.load(open(config.initial_features, 'rb')).A.astype(np.float32)
            self.hidden_size = self.shape[1]
            self.attention_size = config.attention_size
            self.feature_size = features.shape[1]

            with tf.variable_scope('attention'):
                self.features = tf.Variable(features)
                self.w1_attention = tf.Variable(tf.random_normal([self.hidden_size, self.attention_size], stddev=0.1))
                self.w2_attention = tf.Variable(tf.random_normal([self.feature_size, self.attention_size], stddev=0.1))
                self.b_attention = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
                self.v_attention = tf.Variable(tf.random_normal([self.attention_size, 1], stddev=0.1))
                self.bias_mat = tf.Variable(tf.zeros([self.shape[0], self.shape[0]]))

            self.value_mid = tf.tanh(tf.add(tf.matmul(self.embedding_matrix, self.w1_attention), tf.matmul(self.features, self.w2_attention)))
            self.value_mid = tf.add(self.value_mid, self.b_attention)
            self.logits = tf.matmul(self.value_mid, self.v_attention)
            self.logits = self.logits + tf.transpose(self.logits)

            self.coefs = tf.nn.softmax(tf.nn.leaky_relu(self.logits) + self.bias_mat)
            self.z = tf.matmul(self.coefs, self.embedding_matrix)
            # self.z = tf.contrib.layers.bias_add(self.vals)

            self.loss_attention = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.embedding_matrix, logits=self.z)) + config.w_decay * (
                    tf.nn.l2_loss(self.w1_attention) + tf.nn.l2_loss(self.w2_attention) + tf.nn.l2_loss(self.v_attention) + tf.nn.l2_loss(self.b_attention))
        else:
            self.loss_attention = 0

        self.label = tf.placeholder(tf.float32, shape=[None])
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis * (
                            tf.nn.l2_loss(self.node_neighbor_embedding) +
                            tf.nn.l2_loss(self.node_embedding) +
                            tf.nn.l2_loss(self.bias)) + config.lambda_con * tf.reduce_mean(self.difference) + config.w_decay * self.loss_attention  # 最后一项为约束项

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)

        self.c_score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.c_score))

    @staticmethod
    def compute_cosine_similarity(matrix):
        cosine_similarity = tf.constant([], dtype=tf.float32)
        for row in range(matrix.shape[0]):
            mat = tf.tile(tf.expand_dims(matrix[row], axis=0), multiples=[matrix.shape[0] - row, 1])
            cosine_sim = tf.keras.losses.cosine_similarity(mat, matrix[row:])
            cosine_similarity = tf.concat([cosine_similarity, cosine_sim], axis=0)
        return cosine_similarity


class Generator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.all_score = tf.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)  # batch_size * n_embed
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias  # linear mapping
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)  # activate

        if config.constraint:
            self.node_emd_start = tf.constant(self.node_emd_init, dtype=tf.float32)
            # constraint
            self.select = tf.random_shuffle(list(range(self.embedding_matrix.shape[0])))[0:30]
            # the feature distribution before training
            self.emm_start = tf.nn.embedding_lookup(self.node_emd_start, self.select)
            self.emm_start_cosine_similarity = self.compute_cosine_similarity(self.emm_start)
            # after training
            self.emm = tf.nn.embedding_lookup(self.embedding_matrix, self.select)
            self.emm_cosine_similarity = self.compute_cosine_similarity(self.emm)
            # calculate error
            self.difference = tf.abs(self.emm_cosine_similarity - self.emm_start_cosine_similarity)
        else:
            config.lambda_con = 0
            self.difference = tf.constant([0], dtype=tf.float32)  # tf.zeros([0]))  # , trainable=False)

        if config.attention:
            self.shape = self.embedding_matrix.shape

            # load feature
            features = pickle.load(open(config.initial_features, 'rb')).A.astype(np.float32)
            self.hidden_size = self.shape[1]
            self.attention_size = config.attention_size
            self.feature_size = features.shape[1]

            with tf.variable_scope('attention'):
                self.features = tf.Variable(features)
                self.w1_attention = tf.Variable(tf.random_normal([self.hidden_size, self.attention_size], stddev=0.1))
                self.w2_attention = tf.Variable(tf.random_normal([self.feature_size, self.attention_size], stddev=0.1))
                self.b_attention = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
                self.v_attention = tf.Variable(tf.random_normal([self.attention_size, 1], stddev=0.1))
                self.bias_mat = tf.Variable(tf.zeros([self.shape[0], self.shape[0]]))  # bias parameters

            self.value_mid = tf.tanh(tf.add(tf.matmul(self.embedding_matrix, self.w1_attention), tf.matmul(self.features, self.w2_attention)))
            self.value_mid = tf.add(self.value_mid, self.b_attention)
            self.logits = tf.matmul(self.value_mid, self.v_attention)
            self.logits = self.logits + tf.transpose(self.logits)

            self.coefs = tf.nn.softmax(tf.nn.leaky_relu(self.logits) + self.bias_mat)
            self.z = tf.matmul(self.coefs, self.embedding_matrix)
            # self.z = tf.contrib.layers.bias_add(self.vals)

            self.loss_attention = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.embedding_matrix, logits=self.z)) + config.w_decay * (
                    tf.nn.l2_loss(self.w1_attention) + tf.nn.l2_loss(self.w2_attention) + tf.nn.l2_loss(self.v_attention) + tf.nn.l2_loss(self.b_attention))
        else:
            self.loss_attention = 0

        self.reward = tf.placeholder(tf.float32, shape=[None])
        self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + config.lambda_gen * (
                tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding)) + config.lambda_con * tf.reduce_mean(
            self.difference) + config.w_decay * self.loss_attention

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    @staticmethod
    def compute_cosine_similarity(matrix):
        cosine_similarity = tf.constant([], dtype=tf.float32)
        for row in range(matrix.shape[0]):
            mat = tf.tile(tf.expand_dims(matrix[row], axis=0), multiples=[matrix.shape[0] - row, 1])
            cosine_sim = tf.keras.losses.cosine_similarity(mat, matrix[row:])
            cosine_similarity = tf.concat([cosine_similarity, cosine_sim], axis=0)
        return cosine_similarity


class GraphGAN(object):
    def __init__(self):
        print("reading graphs...")
        self.n_node, self.graph = utils.read_edges(config.train_filename, config.test_filename)
        self.n_node = config.shape[0]
        self.root_nodes = [i for i in range(self.n_node)]

        print("reading initial embeddings...")
        self.node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        self.node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        self.node_embed_start = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                      n_node=self.n_node,
                                                      n_embed=config.n_emb)

        # if existing, clean BFS-trees cache
        if os.path.isfile(config.cache_filename):
            print("cleaning cache...")
            os.remove(config.cache_filename)
            print("cache is cleaned!")

        # construct BFS-trees
        print("constructing BFS-trees...")
        self.trees = self.construct_trees(self.root_nodes)
        pickle_file = open(config.cache_filename, 'wb')
        pickle.dump(self.trees, pickle_file)
        pickle_file.close()

        print("building GAN model...")
        self.discriminator = None
        self.attention_discriminator = None
        self.generator = None
        self.attention_generator = None
        self.build_discriminator()
        self.build_generator()

        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

        self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        self.saver = tf.train.Saver()

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                if cur_node not in self.graph.keys():
                    continue
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def build_generator(self):
        with tf.variable_scope("generator"):
            self.generator = Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)

    def build_discriminator(self):
        with tf.variable_scope("discriminator"):
            self.discriminator = Discriminator(n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def train(self):
        # restore the model from the latest checkpoint if exists
        checkpoint = tf.train.get_checkpoint_state(config.model_log)
        if checkpoint and checkpoint.model_checkpoint_path and config.load_model:
            print("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        self.write_embeddings_to_file()
        # gen_auroc, gen_auprc, dis_auroc, dis_auprc = self.evaluation(self)
        # utils.print_info(gen_auroc, gen_auprc, dis_auroc, dis_auprc)

        print("start training...")
        for epoch in range(config.n_epochs):
            # save the model
            if epoch > 0 and epoch % config.save_steps == 0:
                self.saver.save(self.sess, config.model_log + "model.checkpoint")

            # discriminator
            center_nodes, neighbor_nodes, labels, dis_loss = [], [], [], 0
            for d_epoch in range(config.n_epochs_dis):
                if d_epoch % config.dis_interval == 0:
                    center_nodes, neighbor_nodes, labels = self.prepare_data_for_d()
                # training
                _, dis_loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss],
                                            feed_dict={self.discriminator.node_id: np.array(center_nodes),
                                                       self.discriminator.node_neighbor_id: np.array(neighbor_nodes),
                                                       self.discriminator.label: np.array(labels)})

            # generator
            node_1, node_2, gen_loss, reward = [], [], 0, 0
            for g_epoch in range(config.n_epochs_gen):
                if g_epoch % config.gen_interval == 0:
                    node_1, node_2, reward = self.prepare_data_for_g()
                # training
                _, gen_loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                            feed_dict={self.generator.node_id: np.array(node_1),
                                                       self.generator.node_neighbor_id: np.array(node_2),
                                                       self.generator.reward: np.array(reward)})

            self.write_embeddings_to_file()
            gen_auroc, gen_auprc, dis_auroc, dis_auprc = self.evaluation(self)
            utils.print_info(gen_auroc, gen_auprc, dis_auroc, dis_auprc, epoch, dis_loss, gen_loss)
        print("training completes")

    def prepare_data_for_d(self):
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                if i not in self.graph.keys():
                    continue
                pos = self.graph[i]
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        paths = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                sample, paths_from_i = self.sample(i, self.trees[i], config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2, reward

    def sample(self, root, tree, sample_num, for_d):
        all_score = self.sess.run(self.generator.all_score)
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    @staticmethod
    def get_node_pairs_from_path(path):
        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs

    def write_embeddings_to_file(self):
        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n" for emb in embedding_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)

    @staticmethod
    def evaluation(self):
        results = []
        gen_auroc, gen_auprc, dis_auroc, dis_auprc = 0, 0, 0, 0
        for i in range(2):
            lpe = lp.LinkPredictEval(config.emb_filenames[i], config.test_filename, config.test_neg_filename, self.n_node, config.n_emb)
            # lpe = lp.LinkPredictEval(config.emb_filenames[i], config.val_filename, config.val_neg_filename, self.n_node, config.n_emb)
            result = (lpe.eval_link_prediction())
            if i == 0:
                gen_auroc, gen_auprc = result[0], result[1]
            if i == 1:
                dis_auroc, dis_auprc = result[0], result[1]
            results.append("{}: auroc={:.5f}, auprc={:.5f}\n".format(config.modes[i], result[0], result[1]))
        with open(config.result_filename, mode="a+") as f:
            f.writelines(results)
        return gen_auroc, gen_auprc, dis_auroc, dis_auprc
