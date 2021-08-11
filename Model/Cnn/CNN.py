from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import jieba
import pandas as pd
import random
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
stopwords = pd.read_csv("../../stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values
df_list = []
global n_words


def preprocess_text(content_lines, sentences, category) :
    for line in content_lines :
        try :
            segs = jieba.lcut(line[3])
            segs = filter(lambda x : len(x) > 1, segs)
            segs = filter(lambda x : x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except :
            continue


df = pd.read_excel('../../Platform_information/Douban_information/Douban_reviews.xlsx')

df.label.value_counts()
data_com_X_1 = df[df.label == 1]
data_com_X_0 = df[df.label == 0]

# Generate training data
# delete those with null data
sentences = []
preprocess_text(data_com_X_1.values.tolist(), sentences, 'like')
n = 0
while n < 20:
    preprocess_text(data_com_X_0.values.tolist(), sentences, 'nlike')
    n += 1
random.shuffle(sentences)

a =random.shuffle(sentences)
# print(a)
from sklearn.model_selection import train_test_split

x, y = zip(*sentences)
train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)

learn = tf.contrib.learn
FLAGS = None
# Maximum document length
MAX_DOCUMENT_LENGTH = 100
# Minimum word frequency
MIN_WORD_FREQUENCE = 5
# Dimensionality of word embedding
EMBEDDING_SIZE = 20
# Number of filters
N_FILTERS = 10  # 10 neurons
# Perception of wild size
WINDOW_SIZE = 20
# The shape of filter
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
# Pooling
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0


def cnn_model(features, target) :

    """
       A 2-layer convolutional neural network for short text classification
    """
    # First convert words to word embeddings
    # We get a word list mapping matrix of shape [n_words, EMBEDDING_SIZE]
    # Then we can map a batch of text into a matrix of the form [batch_size, sequence_length,EMBEDDING_SIZE]
    target = tf.one_hot(target, 15, 1, 0)
    word_vectors = tf.contrib.layers.embed_sequence(features
                                                    , vocab_size=n_words
                                                    , embed_dim=EMBEDDING_SIZE
                                                    , scope='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1') :
        # Adding a convolution layer for filtering
        conv1 = tf.contrib.layers.convolution2d(word_vectors
                                                , N_FILTERS
                                                , FILTER_SHAPE1
                                                , padding='VALID')
        # Add RELU non-linearity
        conv1 = tf.nn.relu(conv1)
        # Maximum pooling
        pool1 = tf.nn.max_pool(conv1
                               , ksize=[1, POOLING_WINDOW, 1, 1]
                               , strides=[1, POOLING_STRIDE, 1, 1]
                               , padding='SAME')
        # Transpose the matrix to meet the shape
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])

    with tf.variable_scope('CNN_Layer2'):
        # 2nd convolutional layer
        conv2 = tf.contrib.layers.convolution2d(pool1
                                                , N_FILTERS
                                                , FILTER_SHAPE2
                                                , padding='VALID')
        # Extraction characteristics
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # Fully connected layer
    logits = tf.contrib.layers.fully_connected(pool2, 15, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    # Optimizers
    train_op = tf.contrib.layers.optimize_loss(loss
                                               , tf.contrib.framework.get_global_step()
                                               , optimizer='Adam'
                                               , learning_rate=0.01)

    return ({
                'class' : tf.argmax(logits, 1),
                'prob' : tf.nn.softmax(logits)
            }, loss, train_op)


# Handling vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_test = np.array(list(vocab_processor.transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words:%d' % n_words)

cate_dic = {'like' : 1, 'nlike' : 0}
y_train = pd.Series(train_target).apply(lambda x : cate_dic[x], train_target)
y_test = pd.Series(test_target).apply(lambda x : cate_dic[x], test_target)

# Building the model
classifier = learn.SKCompat(learn.Estimator(model_fn=cnn_model))

# Training and Prediction
classifier.fit(x_train, y_train, steps=1000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)


print('Accuracy:{0:f}'.format(score))
a = '%.5f' % score
with open('../../price_information.txt', 'a', encoding='utf-8') as f:
    text = a + '\n'
    f.write(text)