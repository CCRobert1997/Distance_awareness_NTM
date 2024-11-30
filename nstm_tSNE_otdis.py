import math

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
import sys
import os
import time
from scipy import sparse
import scipy.io
#tf.disable_v2_behavior()


base_dir = "./"
sys.path.insert(1, base_dir)

import utils
from auto_diff_sinkhorn import sinkhorn_tf



from utils import load_data, batch_indices, print_topics, set_logger, save_flags, get_doc_topic

tf.disable_v2_behavior()
flags = tf.compat.v1.flags
flags.DEFINE_float('sh_epsilon', 0.001, 'sinkhorn epsilon')
flags.DEFINE_integer('sh_iterations', 50, 'sinkhorn iterations')
flags.DEFINE_string('dataset', 'TMN', 'dataset')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_integer('batch_size', 200, 'batch_size')
flags.DEFINE_integer('K', 100, 'num topics')
flags.DEFINE_integer('random_seed', 1, 'random_seed')
flags.DEFINE_integer('n_epochs', 10, 'n_epochs')#50, 'n_epochs')
flags.DEFINE_float('rec_loss_weight', 0.07, 'rec_loss_weight')
flags.DEFINE_float('tSNE_loss_weight', 1.0, 'tSNE_loss_weight')
flags.DEFINE_float('sh_alpha', 20, 'sh_alpha')
flags.DEFINE_float('perplexity', 30.0, 'perplexity')
flags.DEFINE_integer('X_size_bound', 50, 'X_size_bound')
FLAGS = flags.FLAGS


def sinkhorn_algorithm(dist_1, dist_2, cost_matrix, gamma, num_ot_iter):
    psi_1 = tf.ones_like(dist_1) / tf.cast(tf.shape(dist_1)[-1], tf.float32)
    psi_2 = tf.ones_like(dist_2) / tf.cast(tf.shape(dist_2)[-1], tf.float32)
    cost_gamma = - cost_matrix * gamma
    h = tf.exp(cost_gamma)
    #print(dist_2.shape)
    #print(psi_1.shape)
    #print(h.shape)
    for _ in range(num_ot_iter):
        temp1 = (tf.matmul(psi_1, h) + 1e-12)
        psi_2 = tf.multiply(dist_2, 1 / temp1)
        psi_1 = tf.multiply(dist_1, 1 / (tf.matmul(psi_2, tf.transpose(h)) + 1e-12))
    ot_distances = tf.reduce_sum(tf.multiply(tf.matmul(psi_1, tf.multiply(h, cost_matrix)), psi_2), axis=-1)
    #print("ot_diatance shape is :")
    #print(ot_distances.shape)
    mean_ot_loss = tf.reduce_sum(ot_distances) / tf.cast(tf.shape(dist_1)[0], tf.float32)
    #print(mean_ot_loss)
    return ot_distances, psi_1, psi_2, h, cost_gamma

def seach_prob_with_dist(x, dist, tol=1e-5, perplexity=30.0):
    # '''二分搜索寻找beta,并计算pairwise的prob
    # '''
    # 初始化参数
    print("dist is ")
    print(dist)


    print("Computing pairwise distances...")
    (n, d) = x.shape

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
        perp, this_prob = utils.cal_perplexity(dist[i], i, beta[i])
        print("perb and this prob are:")
        print(perp)
        print(this_prob)
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
            perp, this_prob = utils.cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        print(this_prob)
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    #每个点对其他点的条件概率分布pi\j
    return pair_prob



def run_ntsm(args):


    np.random.seed(FLAGS.random_seed)
    tf.random.set_random_seed(FLAGS.random_seed)

    save_dir = os.path.join(base_dir, 'save', 'dataset%s_K%d_RW%0.3f_RS%d_L%0.3ftSNEW%0.3f' %
                            (FLAGS.dataset, FLAGS.K, FLAGS.rec_loss_weight, FLAGS.random_seed, FLAGS.sh_alpha, FLAGS.tSNE_loss_weight))

    os.makedirs(save_dir, exist_ok=True)

    save_flags(save_dir)

    logger = set_logger(save_dir)

    data_dir = os.path.join(base_dir, 'datasets')
    data_dir = '%s/%s' % (data_dir, FLAGS.dataset)
    #print("111111111111111111111111111111")
    train_data, test_data, word_embeddings, voc = load_data('%s/data.mat' % data_dir, True)

    L = word_embeddings.shape[1]

    V = train_data.shape[1]
    N = train_data.shape[0]
    #print("222222222222222222222222222222")
    ######
    #tSNE



    #tSNE_X = tf.placeholder(name="tSNE_X", dtype=tf.float32, shape=[None, None])
    tSNE_X = tf.placeholder(dtype=tf.float32, shape=[None, None])
    #####
    
    #print("333333333333333333333333333333333333333")
    doc_word_ph = tf.placeholder(dtype=tf.float32, shape=[None, V])

    doc_word_tf = tf.nn.softmax(doc_word_ph)
    #print("44444444444444444444444444444444444")
    with tf.variable_scope('encoder'):

        doc_topic_tf = utils.mlp(doc_word_ph, [200], utils.myrelu)
        doc_topic_tf = tf.nn.dropout(doc_topic_tf, 0.75)
        doc_topic_tf = tf.layers.batch_normalization(utils.linear(doc_topic_tf, FLAGS.K, scope='mean'))
        doc_topic_tf = tf.nn.softmax(doc_topic_tf)

    #print("5555555555555555555555555555555555555555")
    with tf.variable_scope('cost_function'):

        topic_embeddings_tf = tf.get_variable(name='topic_embeddings', shape=[FLAGS.K, L],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, seed=FLAGS.random_seed))
        word_embeddings_ph = tf.placeholder(dtype=tf.float32, shape=[V, L])

        topic_embedding_norm = tf.nn.l2_normalize(topic_embeddings_tf, dim=1)
        word_embedding_norm = tf.nn.l2_normalize(word_embeddings_ph, dim=1)
        topic_word_tf = tf.matmul(topic_embedding_norm, tf.transpose(word_embedding_norm))
        word_word_tf = tf.matmul(word_embedding_norm, tf.transpose(word_embedding_norm))
        M = 1 - topic_word_tf
        M_word = 1 - word_word_tf

    doc_word_dist_repeat = tf.reshape(tf.tile(doc_word_tf, [1, FLAGS.X_size_bound]), [-1, V])
    print("doc topic dist repeat shape is :")
    print(doc_word_dist_repeat.shape)

    doc_word_dist_repeat_compare = tf.reshape(tf.tile(doc_word_tf, [FLAGS.X_size_bound, 1]), [-1, V])
    print("doc topic dist repeat compare shape is :")
    print(doc_word_dist_repeat_compare.shape)

    ot_word_distances, psi_1_word, psi_2_word, h, cost_gamma = sinkhorn_algorithm(doc_word_dist_repeat, doc_word_dist_repeat_compare, M_word, 20, 10)
    ot_word_distances_metric = tf.reshape(ot_word_distances, [-1, FLAGS.X_size_bound])
    print("reshaped ot_distances_metric shape is :")
    print(ot_word_distances_metric.shape)
    ot_word_loss = tf.reduce_mean(tf.reduce_sum(ot_word_distances, axis=-1))







    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    is_stop = False
    with session as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        nb_batches = int(math.ceil(float(N) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= N

        ot_word_losses = []

        running_times = []


        X = train_data[:FLAGS.X_size_bound]
        if sparse.isspmatrix(X):
            X = X.toarray()
        batch_start_time = time.time()
        logger.info('Start solve word ot dist..............')
        ot_word_loss_all, ot_distances_metric_all, ot_word_distances_all, psi_1_word_all, psi_2_word_all, h_all, cost_gamma_all, M_word_all = sess.run([ot_word_loss, ot_word_distances_metric, ot_word_distances, psi_1_word, psi_2_word, h, cost_gamma, M_word], feed_dict={doc_word_ph: X, word_embeddings_ph: word_embeddings})
        logger.info("M_word_all is ")
        print(M_word_all.shape)
        print(M_word_all)
        logger.info("cost_gamma_all is:")
        print(cost_gamma_all)
        logger.info("psi_1_word_all and psi_2_word_all and h_all are:")
        print(h_all)
        print(psi_1_word_all)
        print(psi_2_word_all)
        logger.info("ot_word_distances_all is :")
        print(ot_word_distances_all)
        logger.info('word ot metric ready..................')
        running_times.append(time.time() - batch_start_time)
        logger.info('Solve prob for tSNE...................')
        print(X.shape)
        print(X)
        pair_prob = seach_prob_with_dist(X, ot_distances_metric_all, tol=1e-5, perplexity=30.0)
        logger.info('tSNE prob ready.......................')
        print(pair_prob)







if __name__ == '__main__':

    tf.compat.v1.app.run(run_ntsm)
