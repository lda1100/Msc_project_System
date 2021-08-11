from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
import jieba
import pandas as pd
import random
learn = tf.contrib.learn

FALGS = None

MAX_DOCUMENT_LENGTH = 15
MIN_WORD_FREQUENCE = 1
EMBEDDING_SIZE = 50


global n_words
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
        #             print(sentences)
        except :
            print(line[3])
            continue


df = pd.read_excel('../../Platform_information/Douban_information/Douban_reviews.xlsx')

df.label.value_counts()
data_com_X_1 = df[df.label == 1]
data_com_X_0 = df[df.label == 0]

# Generate training data
# ********* delete those with null data **********
sentences = []
preprocess_text(data_com_X_1.values.tolist(), sentences, 'like')
n = 0
while n < 20 :
    preprocess_text(data_com_X_0.values.tolist(), sentences, 'nlike')
    n += 1
random.shuffle(sentences)

from sklearn.model_selection import train_test_split

x, y = zip(*sentences)
train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH
                                                          , min_frequency=MIN_WORD_FREQUENCE)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_test = np.array(list(vocab_processor.transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)


def rnn_model(features, target) :
    """
    GRU
    """
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size,sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(features
                                                    , vocab_size=n_words
                                                    , embed_dim=EMBEDDING_SIZE
                                                    , scope='words')

    # Split into list of embedding per word, while removing doc length dim。
    # word_list results to be a list of tensors [batch_size,EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    target = tf.one_hot(target, 15, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
    # Create cross-entropy loss
    # logits is [batch_size, sequence_length] The output of the logging network. target is [batch_size, sequence_length] Single heat encoded label.
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)

    return ({
                'class' : tf.argmax(logits, 1),
                'prob' : tf.nn.softmax(logits)
            }, loss, train_op)


model_fn = rnn_model
classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))
cate_dic={'like':1,'nlike':0}
y_train = pd.Series(train_target).apply(lambda x:cate_dic[x] , train_target)
y_test = pd.Series(test_target).apply(lambda x:cate_dic[x] , test_target)
# Train and predict
classifier.fit(x_train, y_train, steps=1000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy:{0:f}'.format(score))
a = '%.5f' % score
with open('../../price_information.txt', 'a', encoding='utf-8') as f:
    text = a + '\n'
    f.write(text)