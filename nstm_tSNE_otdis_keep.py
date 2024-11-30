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
flags.DEFINE_integer('batch_size', 100, 'batch_size')
flags.DEFINE_integer('K', 100, 'num topics')
flags.DEFINE_integer('random_seed', 1, 'random_seed')
flags.DEFINE_integer('n_epochs', 10, 'n_epochs')#50, 'n_epochs')
flags.DEFINE_float('rec_loss_weight', 0.07, 'rec_loss_weight')
flags.DEFINE_float('tSNE_loss_weight', 1.0, 'tSNE_loss_weight')
flags.DEFINE_float('sh_alpha', 20, 'sh_alpha')
flags.DEFINE_float('perplexity', 30.0, 'perplexity')
FLAGS = flags.FLAGS


def sinkhorn_algorithm(dist_1, dist_2, cost_matrix, gamma, num_ot_iter):
    psi_1 = tf.ones_like(dist_1) / tf.cast(tf.shape(dist_1)[-1], tf.float32)
    psi_2 = tf.ones_like(dist_2) / tf.cast(tf.shape(dist_2)[-1], tf.float32)
    h = tf.exp(- cost_matrix * gamma)
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
    return ot_distances

def seach_prob_with_dist(x, dist, tol=1e-5, perplexity=30.0):
    # '''二分搜索寻找beta,并计算pairwise的prob
    # '''
    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    #dist = cal_pairwise_dist(x)
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
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    #每个点对其他点的条件概率分布pi\j
    return pair_prob



def run_ntsm(args):


    np.random.seed(FLAGS.random_seed)
    tf.random.set_random_seed(FLAGS.random_seed)

    save_dir = os.path.join(base_dir, 'save', 'dataset%s_K%d_RW%0.3f_RS%d_L%0.3ftSNE_word_OT_W%0.3f' %
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

    doc_word_dist_repeat = tf.reshape(tf.tile(doc_word_tf, [1, FLAGS.batch_size]), [-1, V])
    print("doc topic dist repeat shape is :")
    print(doc_word_dist_repeat.shape)

    doc_word_dist_repeat_compare = tf.reshape(tf.tile(doc_word_tf, [FLAGS.batch_size, 1]), [-1, V])
    print("doc topic dist repeat compare shape is :")
    print(doc_word_dist_repeat_compare.shape)

    ot_word_distances = sinkhorn_algorithm(doc_word_dist_repeat, doc_word_dist_repeat_compare, M_word, 20, 10)
    ot_word_distances_metric = tf.reshape(ot_word_distances, [-1, FLAGS.batch_size])
    print("reshaped ot_distances_metric shape is :")
    print(ot_word_distances_metric.shape)
    ot_word_loss = tf.reduce_mean(tf.reduce_sum(ot_word_distances, axis=-1))

    sh_loss = sinkhorn_tf(M, tf.transpose(doc_topic_tf), tf.transpose(doc_word_tf), lambda_sh=FLAGS.sh_alpha)

    sh_loss = tf.reduce_mean(sh_loss)

    # print("7777777777777777777777777777777777777777")
    rec_log_probs = tf.nn.log_softmax(tf.matmul(doc_topic_tf, topic_word_tf))
    # rec_log_probs = tf.nn.log_softmax(tf.matmul(tf.nn.softmax(doc_topic_tf), 2 - M))
    rec_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(rec_log_probs, doc_word_ph), 1))

    fullvars = tf.trainable_variables()
    # print("????????????????????????????????????????")
    print(fullvars)
    enc_vars = utils.variable_parser(fullvars, 'encoder')
    cost_function_vars = utils.variable_parser(fullvars, 'cost_function')

    rec_loss_weight = tf.placeholder(tf.float32, ())
    tSNE_loss_weight = tf.placeholder(tf.float32, ())

    ##############
    # tSNE
    sum_y = tf.reduce_sum(tf.square(doc_topic_tf), 1)
    temp = tf.add(tf.transpose(tf.add(-2 * tf.matmul(doc_topic_tf, tf.transpose(doc_topic_tf)), sum_y)), sum_y)
    num = tf.divide(1, 1 + temp)
    one_ = tf.constant([x for x in range(FLAGS.batch_size)])
    one_hot = tf.one_hot(one_, FLAGS.batch_size)
    num = num - num * one_hot
    Q = num / tf.reduce_sum(num)
    Q = tf.maximum(Q, 1e-12)
    # print("Q shape is :")
    # print(Q.shape)

    # learning_rate = 500
    tSNE_loss = tf.reduce_sum(tSNE_X * tf.log(tf.divide(tSNE_X, Q)))
    ######

    joint_loss = rec_loss_weight * rec_loss + sh_loss + tSNE_loss_weight * tSNE_loss
    trainer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(joint_loss, var_list=[enc_vars + cost_function_vars])

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    is_stop = False
    with session as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        nb_batches = int(math.ceil(float(N) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= N

        rec_losses = []

        sh_losses = []

        joint_losses = []

        ot_word_losses = []

        running_times = []

        for epoch in range(FLAGS.n_epochs):

            logger.info('epoch: %d' % epoch)

            idxlist = np.random.permutation(N)

            rec_loss_avg, sh_loss_avg, joint_loss_avg, ot_word_loss_avg = 0., 0., 0., 0.

            for batch in tqdm(range(nb_batches)):

                start, end = batch_indices(batch, N, FLAGS.batch_size)
                X = train_data[idxlist[start:end]]



                if sparse.isspmatrix(X):
                    X = X.toarray()
                batch_start_time = time.time()
                ot_word_loss_batch, ot_distances_metric_batch = sess.run([ot_word_loss, ot_word_distances_metric], feed_dict={doc_word_ph: X, word_embeddings_ph: word_embeddings})
                running_times.append(time.time() - batch_start_time)

                pair_prob = seach_prob_with_dist(X, ot_distances_metric_batch, tol=1e-5, perplexity=30.0)

                P = pair_prob + np.transpose(pair_prob)
                P = P / np.sum(P)  # pij
                P = np.maximum(P, 1e-12)

                print(pair_prob)
                print(P)



                _, rec_loss_batch, sh_rec_loss_batch, tSNE_loss_batch, joint_loss_batch = \
                    sess.run([trainer, rec_loss, sh_loss, tSNE_loss, joint_loss],
                             feed_dict={doc_word_ph: X, word_embeddings_ph: word_embeddings, tSNE_X: P,
                                        rec_loss_weight: FLAGS.rec_loss_weight,
                                        tSNE_loss_weight: FLAGS.tSNE_loss_weight})

                print(tSNE_loss_batch)
                running_times.append(time.time() - batch_start_time)
                if np.isnan(joint_loss_batch):
                    is_stop = True

                rec_loss_avg += rec_loss_batch
                sh_loss_avg += sh_rec_loss_batch
                joint_loss_avg += joint_loss_batch
                ot_word_loss_avg += ot_word_loss_batch

                rec_losses.append(rec_loss_batch)
                sh_losses.append(sh_rec_loss_batch)
                joint_losses.append(joint_loss_batch)
                ot_word_losses.append(ot_word_loss_batch)


            logger.info('joint_loss: %f' % (joint_loss_avg / nb_batches))

            if is_stop:
                logger.info('early stop because of NaN at epoch %d' % epoch)
                break

            if (epoch%10 == 0):

                [topic_embeddings, topic_word_mat] = sess.run([topic_embeddings_tf, topic_word_tf], feed_dict={word_embeddings_ph: word_embeddings})

                train_doc_topic = get_doc_topic(sess, doc_topic_tf, doc_word_ph, train_data, FLAGS.K)

                test_doc_topic = get_doc_topic(sess, doc_topic_tf, doc_word_ph, test_data, FLAGS.K)

                vis_file = open(os.path.join(save_dir, 'vis.txt'), 'a')
                print_topics(topic_word_mat, voc, printer=vis_file.write)


                scipy.io.savemat(os.path.join(save_dir, 'save.mat'), {'phi': topic_word_mat,
                                                                        'train_theta': train_doc_topic,
                                                                        'test_theta': test_doc_topic,
                                                                        'topic_embeddings': topic_embeddings,
                                                                        'joint_losses': joint_losses})
                logger.info('data saved at epoch %d' % epoch)








if __name__ == '__main__':

    tf.compat.v1.app.run(run_ntsm)
