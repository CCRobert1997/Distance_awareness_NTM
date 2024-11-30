#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import json
import logging
import os
import scipy.io as sio
from scipy import sparse
import scipy.stats
import math
from operator import itemgetter
from sklearn import neighbors
from scipy import linalg
from math import log2

tf.disable_v2_behavior()


def topic_word_ot(self):
    doc_topic_dist_repeat = tf.reshape(tf.tile(self.doc_topic_dist, [1, self.num_neighbors]), [-1, self.num_topics])
    minibatch_neighboring_documents_reshape = tf.reshape(self.minibatch_neighboring_documents, [-1, self.num_words])
    minibatch_neighboring_documents_norm = tf.nn.softmax(minibatch_neighboring_documents_reshape)
    self.evaluate_topic_word_cost_matrix()
    ot_distances = self.sinkhorn_algorithm(doc_topic_dist_repeat, minibatch_neighboring_documents_norm,
                                           self.topic_word_cost_matrix)
    ot_distances = tf.reshape(ot_distances, [-1, self.num_neighbors])
    ot_distances = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.attention_values, ot_distances), axis=-1))

    return ot_distances


def topic_structure_ot(self):
    doc_topic_dist_repeat = tf.reshape(tf.tile(self.doc_topic_dist, [1, self.num_neighbors]), [-1, self.num_topics])
    minibatch_neighboring_adjacency_reshape = tf.reshape(self.minibatch_neighboring_adjacency,
                                                         [-1, self.num_training_documents])
    minibatch_neighboring_adjacency_norm = tf.nn.softmax(minibatch_neighboring_adjacency_reshape)
    self.evaluate_topic_structure_cost_matrix()
    ot_distances = self.sinkhorn_algorithm(doc_topic_dist_repeat, minibatch_neighboring_adjacency_norm,
                                           self.topic_structure_cost_matrix)
    ot_distances = tf.reshape(ot_distances, [-1, self.num_neighbors])
    ot_distances = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.attention_values, ot_distances), axis=-1))

    return ot_distances


def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


def hellinger_distance(p, q):
    h1 = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
    #or h2 = np.sqrt(1 - np.sum(np.sqrt(p * q)))
    return h1

def bhattacharyya_distance(p, q):
    BC = np.sum(np.sqrt(p * q))
    b = -np.log(BC)
    return b

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)



#for LLE
class LLE:
    def __init__(self, X, K):
        self.K = K
        self.X = X

    def Knbor_Mat(self, dist_metric="euclidean", algorithm="ball_tree"):

        K = self.K
        X = self.X
        """
        def mydist(x, y):
            return np.sum((x - y) ** 2)
        metric = 'pyfunc', func = mydist)
        """
        knn = neighbors.NearestNeighbors(n_neighbors=K + 1, metric=dist_metric, algorithm=algorithm).fit(X)
        distances, nbors = knn.kneighbors(X)
        return (nbors[:, 1:])

    def get_weights(self, reg):

        K = self.K
        X = self.X

        nbors = self.Knbor_Mat()
        n, _ = self.X.shape
        Weights = np.zeros((n, n))
        for i in range(n):

            X_bors = X[nbors[i], :] - X[i]
            cov_nbors = np.dot(X_bors, X_bors.T)
            # regularization tems
            trace = np.trace(cov_nbors)
            if trace > 0:
                R = reg * trace
            else:
                R = reg
            cov_nbors.flat[::K + 1] += R
            weights = linalg.solve(cov_nbors, np.ones(K).T, sym_pos=True)
            weights = weights / weights.sum()
            Weights[i, nbors[i]] = weights

        return (Weights)


#for tSNE calculate dist
def cal_pairwise_dist(x):
    # '''计算pairwise 距离, x是matrix
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    # '''
    sum_x = np.sum(np.square(x), 1)
    # print -2 * np.dot(x, x.T)
    # print np.add(-2 * np.dot(x, x.T), sum_x).T
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

#for tSNE calculate cal_perplexity
def cal_perplexity(dist, idx=0, beta=1.0):
    # '''计算perplexity, D是距离向量，
    # idx指dist中自己与自己距离的位置，beta是高斯分布参数
    # 这里的perp仅计算了熵，方便计算
    # '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob == 0:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
        prob /= sum_prob
    #困惑度和pi\j的概率分布
    return perp, prob

#for tSNE calculate seach_prob
def seach_prob(x, tol=1e-5, perplexity=30.0, dist_type="eclidean", load_pairwise_dist=None):
    # '''二分搜索寻找beta,并计算pairwise的prob
    # '''
    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    if (dist_type=="eclidean"):
        dist = cal_pairwise_dist(x)
    elif (dist_type=="ot"):
        dist = np.transpose(load_pairwise_dist) + load_pairwise_dist
    else:
        dist = cal_pairwise_dist(x)
    print(load_pairwise_dist)
    print(dist)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." %(i,n))

        betamin = -np.inf
        betamax = np.inf
        #dist[i]需要换不能是所有点
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    #每个点对其他点的条件概率分布pi\j
    return pair_prob



def load_data(mat_file_name, is_to_dense=True):

    data = sio.loadmat(mat_file_name)
    print(123456)
    train_data = data['wordsTrain'].transpose()
    test_data = data['wordsTest'].transpose()
    print(67890)
    if (train_data.shape[0] > 10000):
        intervalselect = int(train_data.shape[0]/10000)
        select_row = [intervalselect*i for i in range(10000)]
        train_data = train_data[select_row,:]
    word_embeddings = data['embeddings']
    voc = data['vocabulary']
    print(12345)

    voc = [v[0][0] for v in voc]
    print(67890)
    if is_to_dense:
        print(11111)
        if sparse.isspmatrix(train_data):
            print(222222)
            train_data = train_data.toarray()
        print(333333)
        print(train_data.shape)
        train_data = train_data.astype('float32')
        print(444444)
        if sparse.isspmatrix(test_data):
            test_data = test_data.toarray()
        print(5555555)
        test_data = test_data.astype('float32')

    return train_data, test_data, word_embeddings, voc



def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
    """Define a linear connection."""
    with tf.variable_scope(scope or 'Linear'):
        if matrix_start_zero:
            matrix_initializer = tf.constant_initializer(0)
        else:
            matrix_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        if bias_start_zero:
            bias_initializer = tf.constant_initializer(0)
        else:
            bias_initializer = None
        input_size = inputs.get_shape()[1].value

        if weights is not None:
            matrix = weights
        else:
            matrix = tf.get_variable('Matrix', [input_size, output_size], initializer=matrix_initializer)

        output = tf.matmul(inputs, matrix)
        if not no_bias:
            bias_term = tf.get_variable('Bias', [output_size],
                                        initializer=bias_initializer)
            output = output + bias_term
    return output


def mlp(inputs,
        mlp_hidden=[],
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
    """Define an MLP."""
    with tf.variable_scope(scope or 'Linear'):
        mlp_layer = len(mlp_hidden)
        res = inputs
        for l in range(mlp_layer):
            res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l' + str(l)))
        return res

def myrelu(features):
    return tf.maximum(features, 0.0)

def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end



def set_logger(save_dir=None, logger_name='nstm', log_file_name='log.txt'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, log_file_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def save_flags(results_dir):
    FLAGS = tf.compat.v1.flags.FLAGS
    train_params = json.dumps({k: v.value
                               for k, v in FLAGS._flags().items()}, sort_keys=True)
    with open(os.path.join(results_dir, 'params.txt'), 'a') as f:
        f.writelines(str(train_params))
        f.write('\n')


def get_doc_topic(sess, doc_topic_tf, doc_word_tf, doc_word, K, other_param_tf=None, batch_size=200):
    N = np.shape(doc_word)[0]
    nb_batches = int(math.ceil(float(N) / batch_size))
    assert nb_batches * batch_size >= N

    import scipy.sparse

    doc_topic = np.zeros((N, K))
    for batch in range(nb_batches):
        start, end = batch_indices(batch, N, batch_size)
        X = doc_word[start:end]

        if scipy.sparse.issparse(X):
            X = X.todense()
            X = X.astype('float32')

        feed_dict = {doc_word_tf: X}
        if other_param_tf is not None:
            feed_dict.update(other_param_tf)

        temp = sess.run(doc_topic_tf, feed_dict)

        doc_topic[start:end] = temp

    return doc_topic

def print_topics(topic_word_mat, voc, doc_topic_mat=None, sample_doc_word_mat=None, top_words_N=10, top_docs_N=2,
                 printer=None):
    if printer == None:
        printer = print

    K = np.shape(topic_word_mat)[0]

    V = np.shape(topic_word_mat)[1]

    if doc_topic_mat is not None:
        rank = np.argsort(np.sum(doc_topic_mat, axis=0))[::-1]
    else:
        rank = list(range(K))

    assert V == len(voc)

    for k in rank:

        top_word_idx = np.argsort(topic_word_mat[k, :])[::-1]

        top_word_idx = top_word_idx[0:top_words_N]

        top_words = itemgetter(*top_word_idx)(voc)

        printer('topic %d: [%s]\n' % (k, ', '.join(map(str, top_words))))

        if doc_topic_mat is not None:

            doc_rank = np.argsort(doc_topic_mat[:, k])[::-1]

            doc_rank = doc_rank[0:top_docs_N]

            for i in doc_rank:
                doc_words_idx = np.nonzero(sample_doc_word_mat[i, :])[0]

                top_words = itemgetter(*doc_words_idx)(voc)

                printer('*******doc words: [%s]' % ', '.join(map(str, top_words)))
    printer('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
        elif prefix in varname:
            ret_list.append(var)
    return ret_list
