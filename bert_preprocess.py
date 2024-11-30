import numpy as np
import json
import logging
import os
import scipy.io as sio
from scipy import sparse
import math
import tensorflow_text as text  # Registers the ops.
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.datasets import fetch_20newsgroups
import pickle

newsgroups_train = fetch_20newsgroups(subset='train')






text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1")
encoder_inputs = preprocessor(text_input) # dict with keys: 'input_mask', 'input_type_ids', 'input_word_ids'
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]      # [batch_size, 768].
sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

embedding_model = tf.keras.Model(text_input, pooled_output)
sentences = tf.constant(newsgroups_train.data)


bert_embedding_batch_size = 10
print(sentences.shape[0])
Number_of_batch_for_bertembedding = int(sentences.shape[0]/bert_embedding_batch_size)
i = 0
train_data = np.empty((0, 768))
print(train_data.shape)
while (i*bert_embedding_batch_size + bert_embedding_batch_size < sentences.shape[0]):
    append_array = embedding_model(sentences[i*bert_embedding_batch_size:i*bert_embedding_batch_size + bert_embedding_batch_size]).numpy()
    train_data = np.concatenate((train_data, append_array), axis=0)
    print(train_data.shape)
    i += 1
train_data = np.concatenate((train_data, embedding_model(sentences[i*bert_embedding_batch_size:]).numpy()), axis=0)
print(train_data.shape)



with open('datasets/20News/train_bert_embedding.pickle', 'wb') as handle:
    pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)