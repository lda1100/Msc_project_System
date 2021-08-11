import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import word2vec
import jieba
import tensorflow as tf
import numpy as np
import time
from random import randint
from random import shuffle


def makeStopWord():
    with open('../../stopwords.txt', 'r', encoding ='utf-8') as f:
        lines = f.readlines()
    stopWord = []
    for line in lines:
        words = jieba.lcut(line,cut_all = False)
        for word in words:
            stopWord.append(word)
    return stopWord

# Convert evaluation data to a matrix, return type array
def words2Array(lineList):
    linesArray=[]
    wordsArray=[]
    steps = []
    # Iterate through each row
    for line in lineList:
        t = 0
        p = 0
        for i in range(MAX_SIZE):
            if i < len(line):
                try:
                    wordsArray.append(model.wv.word_vec(line[i]))
                    p = p + 1
                except KeyError:
                    t = t+1
                    continue
            else:
               wordsArray.append(np.array([0.0]*dimsh))

        for i in range(t):
            wordsArray.append(np.array([0.0]*dimsh))
        steps.append(p)
        linesArray.append(wordsArray)
        wordsArray = []
    linesArray = np.array(linesArray)
    steps = np.array(steps)
    return linesArray, steps

def convert2Data(posArray, negArray, posStep, negStep):
    randIt = []
    data = []
    steps = []
    labels = []
    for i in range(len(posArray)):
        randIt.append([posArray[i], posStep[i], [1,0]])
    for i in range(len(negArray)):
        randIt.append([negArray[i], negStep[i], [0,1]])
    shuffle(randIt)
    for i in range(len(randIt)):
        data.append(randIt[i][0])
        steps.append(randIt[i][1])
        labels.append(randIt[i][2])
    data = np.array(data)
    steps = np.array(steps)
    return data, steps, labels

# Get the information in the txt file and that's per line
def getWords(file):
    wordList = []
    lineList = []
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        # Perform split word cut_all = False Exact mode
        trans = jieba.lcut(line.replace('\n',''), cut_all=False)
        for word in trans:
            if word not in stopWord:
                wordList.append(word)
        lineList.append(wordList)
        wordList = []
    # print(lineList)
    return lineList

def makeData(posPath,negPath):
    # Get vocabulary, return type [[] ,[] ,]
    pos = getWords(posPath)
    print("The positive data's length is :",len(pos))
    neg = getWords(negPath)
    print("The negative data's length is :",len(neg))
    # Convert evaluation data to a matrix, return type array
    posArray, posSteps = words2Array(pos)
    negArray, negSteps = words2Array(neg)
    # Mix positive and negative data together and break them up to make a data set
    Data, Steps, Labels = convert2Data(posArray, negArray, posSteps, negSteps)
    # Data is the matrix converted from evaluation and word vectors, Steps is the length of each corresponding evaluation and Labels is the corresponding evaluation label
    return Data, Steps, Labels


# print()
word2vec_path = 'word2vec/200/word2vec.model'
model = gensim.models.Word2Vec.load(word2vec_path)
# dimsh is the dimensionality of the word vector
dimsh = model.vector_size
# When feeding data to the neural network, the consistency of the data SHAPE should be maintained MAX_SIZE maximum vocabulary converted to matrix
MAX_SIZE=30
stopWord = makeStopWord()

print("In train data:")
trainData, trainSteps, trainLabels = makeData('data/A/Pos-train.txt',
                                              'data/A/Neg-train.txt')
print("In test data:")
testData, testSteps, testLabels = makeData('data/A/Pos-price_information.txt',
                                           'data/A/Neg-price_information.txt')
trainLabels = np.array(trainLabels)

del model


num_nodes = 256
# Batch size
batch_size = 16
# Output size
output_size = 2

# Define a calculation chart
graph = tf.Graph()
# Construction of computational diagrams
with graph.as_default():

    # Placeholder in the model when the neural network builds the graph
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,MAX_SIZE,dimsh))
    tf_train_steps = tf.placeholder(tf.int32,shape=(batch_size))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,output_size))

    # The test data are in the form of constants within the calculation chart
    tf_test_dataset = tf.constant(testData,tf.float32)
    tf_test_steps = tf.constant(testSteps,tf.int32)


    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = num_nodes, state_is_tuple=True)

    # shape, the dimension of the generated tensor, stddev, the standard deviation
    # Need to define its output dimension
    w1 = tf.Variable(tf.truncated_normal([num_nodes,num_nodes // 2], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([num_nodes // 2], stddev=0.1))

    w2 = tf.Variable(tf.truncated_normal([num_nodes // 2, 2], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1))
    
    def model(dataset, steps):

        outputs, last_states = tf.nn.dynamic_rnn(cell = lstm_cell,
                                                 dtype = tf.float32,
                                                 sequence_length = steps,
                                                 inputs = dataset)
        hidden = last_states[-1]

        # (A,B) Matrix A and matrix B multiplied together
        hidden = tf.matmul(hidden, w1) + b1
        logits = tf.matmul(hidden, w2) + b2
        # Only the last value is taken as output
        return logits

    train_logits = model(tf_train_dataset, tf_train_steps)
    # Calculate the value of loss
    loss = tf.reduce_mean(
        # Directly done with softmax and calculating cross entropy
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                logits=train_logits))
    # Optimizer, optimization objective is loss, learning rate is 0.2
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

    # Predictions on the test set
    test_prediction = tf.nn.softmax(model(tf_test_dataset, tf_test_steps))

# The calculation diagram is basically built, the next step is the process of calling
num_steps = 15000
# num_steps = 1000
summary_frequency = 100


with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        offset = (step * batch_size) % (len(trainLabels)-batch_size)
        feed_dict={tf_train_dataset:trainData[offset:offset + batch_size],
                   tf_train_labels:trainLabels[offset:offset + batch_size],
                   tf_train_steps:trainSteps[offset:offset + batch_size]}
        _, l = session.run([optimizer,loss],
                           feed_dict = feed_dict)
        mean_loss += l
        if step >0 and step % summary_frequency == 0:
            mean_loss = mean_loss / summary_frequency
            print("The step is: %d"%(step))
            print("In train data,the loss is:%.4f"%(mean_loss))
            mean_loss = 0
            acrc = 0
            prediction = session.run(test_prediction)
            for i in range(len(prediction)):
                if prediction[i][testLabels[i].index(1)] > 0.5:
                    acrc = acrc + 1
            print("In test data,the accuracy is:%.2f%%"%((acrc/len(testLabels))*100))
    a = str(acrc/len(testLabels))
    # print(int(acrc/len(testLabels)))
    with open('../../price_information.txt', 'a', encoding='utf-8') as f :
        text = a + '\n'
        f.write(text)


