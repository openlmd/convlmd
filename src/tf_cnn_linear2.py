


import numpy as np
import tensorflow as tf


n_input = 784
n_classes = 2
dropout = 0.75
learning_rate = 0.001

def conv2d_layer(input,shape,pool):

    initializer = tf.truncated_normal(shape, stddev=0.1)
    weights = tf.Variable(initializer,name='Weights')
    tf.summary.histogram('Weights', weights)

    initializer = tf.constant(0.1,shape=[shape[3]])
    biases = tf.Variable(initializer,name='Biases')

    tf.summary.histogram('Biases', biases)

    conv = tf.nn.conv2d(input,weights,[1,1,1,1],padding='SAME',name='Conv2d')+biases
    pool = tf.nn.max_pool(conv,[1,pool,pool,1],[1,pool,pool,1],padding='SAME',name='Pool')
    relu = tf.nn.relu(pool,name='ReLU')

    return relu

def fc_layer(input,shape):

    initializer = tf.truncated_normal(shape, stddev=0.1)
    weights = tf.Variable(initializer, name='Weights')
    tf.summary.histogram('Weights',weights)

    initializer = tf.constant(0.1, shape=[shape[1]])
    biases = tf.Variable(initializer, name='Biases')
    tf.summary.histogram('Biases',biases)


    fc = tf.matmul(input,weights)+biases

    return fc



def model_speed(x, y, dropout):


    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.variable_scope('Conv1'):
        conv1 = conv2d_layer(tf.reshape(x, shape=[-1, 28, 28, 1]), [5, 5, 1, 32], 2)

    with tf.variable_scope('Conv2'):
        conv2 = conv2d_layer(conv1, [5, 5, 32, 64], 2)

    with tf.variable_scope('FC1'):
        fc1 = tf.nn.relu(fc_layer(tf.reshape(conv2, shape=[-1, 7 * 7 * 64]), [7 * 7 * 64, 2048]), name='ReLU')

    with tf.variable_scope('Dropout'):
        fc1_drop = tf.nn.dropout(fc1, dropout)

    with tf.variable_scope('FC2'):
        fc2 = tf.nn.relu(fc_layer(fc1_drop, [2048, n_classes]))


    return fc2



def model(x, y, dropout):


    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.variable_scope('Conv1'):
        conv1 = conv2d_layer(tf.reshape(x, shape=[-1, 28, 28, 1]), [5, 5, 1, 32], 2)

    with tf.variable_scope('Conv2'):
        conv2 = conv2d_layer(conv1, [5, 5, 32, 64], 2)

    with tf.variable_scope('FC1'):
        fc1 = tf.nn.relu(fc_layer(tf.reshape(conv2, shape=[-1, 7 * 7 * 64]), [7 * 7 * 64, 1024]), name='ReLU')

    with tf.variable_scope('Dropout'):
        fc1_drop = tf.nn.dropout(fc1, dropout)

    with tf.variable_scope('FC2'):
        fc2 = fc_layer(fc1_drop, [1024, n_classes])


    return fc2

def loss(pred, y):

    with tf.variable_scope('Loss'):
        cost = tf.reduce_mean(tf.square(abs[:,0]-y[:,0]) + tf.square(pred[:,1]-y[:,1]))
        #cost = tf.reduce_mean(tf.abs(pred-y))# * tf.cast(tf.greater(tf.reduce_max(tf.abs(pred - y)),1.),tf.float32)
        #cost = tf.reduce_mean(tf.abs(pred[:,1] - y[:,1]))
        tf.summary.scalar('Loss', cost)

    return cost


def training(pred, y,this_learning_rate=None):

    if this_learning_rate is None:
        this_learning_rate = learning_rate


    cross_entropy = tf.reduce_mean(tf.square(pred[:,0]-y[:,0]) + tf.square(pred[:,1]-y[:,1]))
    #cross_entropy = tf.reduce_mean(tf.abs(pred - y))# * tf.cast(tf.greater(tf.reduce_max(tf.abs(pred - y)),1.),tf.float32)
    #cross_entropy = tf.reduce_mean(tf.abs(pred[:,1] - y[:,1]))+tf.reduce_mean(tf.abs(pred[:,0] - y[:,0]))*0.1# * tf.cast(tf.greater(tf.reduce_max(tf.abs(pred - y)),1.),tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    return cross_entropy, optimizer

def training_fc2(pred, y, this_learning_rate=None):

    if this_learning_rate is None:
        this_learning_rate = learning_rate

    cross_entropy = tf.reduce_mean(tf.square(pred[:,0]-y[:,0]) + tf.square(pred[:,1]-y[:,1]))
    #cross_entropy = tf.reduce_mean(tf.abs(pred - y))# * tf.cast(tf.greater(tf.reduce_max(tf.abs(pred - y)),1.),tf.float32)
    #cross_entropy = tf.reduce_mean(tf.abs(pred[:,1] - y[:,1]))+tf.reduce_mean(tf.abs(pred[:,0] - y[:,0]))*0.1# * tf.cast(tf.greater(tf.reduce_max(tf.abs(pred - y)),1.),tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=this_learning_rate).\
        minimize(cross_entropy,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='FC2'))
    return cross_entropy, optimizer




def evaluation(pred, y):

    with tf.variable_scope('Evaluation'):
        correct_pred = tf.less(tf.reduce_max(tf.abs(pred - y), axis=1), 1.)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    return accuracy


def power_evaluation(pred, y):

    with tf.variable_scope('Evaluation'):
        correct_pred = tf.less(tf.abs(pred[:,0] - y[:,0]), 1.)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    return accuracy


def speed_evaluation(pred, y):

    with tf.variable_scope('Evaluation'):
        correct_pred = tf.less(tf.abs(pred[:,1] - y[:,1]), 1.)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    return accuracy
