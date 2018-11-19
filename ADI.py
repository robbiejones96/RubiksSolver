import py222
from random import randint
import numpy as np
import tensorflow as tf

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

kLearningRate = 0.75

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

# TODO: Add the loss weight to each sample?
def generateSamples(k, l):
    N = k * l
    samples = [None] * N
    for i in range(l):
        currentCube = py222.initState()
        for j in range(k):
            scrambledCube = py222.doAlgStr(currentCube, getRandomMove())
            samples[k * i + j] = scrambledCube
            currentCube = scrambledCube
    return samples

def reward(cube):
    return 1 if py222.isSolved(cube, True) else -1

def forwardPass(cube, sess):
    state = py222.getState(cube).flatten()
    inputState = np.array([state])
    policy = sess.run(policyOutput, feed_dict={X: inputState})
    policy = tf.nn.softmax(policy)
    value = sess.run(valueOutput, feed_dict={X: inputState})
    return value, policy

def train(states, optimalVals, optimalPols, sess):
    optimalVals = np.array([optimalVals]).T
    sess.run(optimizer, feed_dict={X : states, optimalPolicies : optimalPols, optimalValues : optimalVals})

def constructGraph():
    global X
    global optimalPolicies
    global optimalValues
    global layer1
    global layer2
    global policyLayer1
    global valueLayer1
    global policyOutput
    global valueOutput
    global policyLoss
    global valueLoss
    global optimizer

    X = tf.placeholder(tf.float32, shape=[None, 8 * 24], name="Cube")
    layer1 = tf.contrib.layers.fully_connected(X, 4096, tf.nn.elu, weights_initializer=tf.glorot_uniform_initializer)
    layer2 = tf.contrib.layers.fully_connected(layer1, 2048, tf.nn.elu, weights_initializer=tf.glorot_uniform_initializer)
    policyLayer1 = tf.contrib.layers.fully_connected(layer2, 512, tf.nn.elu, weights_initializer=tf.glorot_uniform_initializer)
    valueLayer1 = tf.contrib.layers.fully_connected(layer2, 512, tf.nn.elu, weights_initializer=tf.glorot_uniform_initializer)
    policyOutput = tf.contrib.layers.fully_connected(policyLayer1, 12, activation_fn=None, weights_initializer=tf.glorot_uniform_initializer)
    valueOutput = tf.contrib.layers.fully_connected(valueLayer1, 1, activation_fn=None, weights_initializer=tf.glorot_uniform_initializer)
    optimalPolicies = tf.placeholder(tf.float32, shape=[None, 12], name="optimalPolicies")
    optimalValues = tf.placeholder(tf.float32, shape=[None, 1], name="OptimalValues")
    policyLoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=optimalPolicies, logits=policyOutput)
    valueLoss = tf.losses.mean_squared_error(optimalValues, valueOutput)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=kLearningRate).minimize(valueLoss + policyLoss)


# TODO: might need to change numpy arrays to TF variables
def doADI(k, l, M):
    constructGraph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(M):
        samples = generateSamples(k, l)
        states = np.empty((len(samples), 8 * 24))
        optimalVals = np.empty(len(samples))
        optimalPolicies = np.empty((len(samples), len(moves)))  
        for i, sample in enumerate(samples):
            values = np.empty(len(moves))
            for j, move in enumerate(moves):
                child = py222.doAlgStr(sample, move)
                value, _ = forwardPass(child, sess)
                values[j] = value + reward(child)
            optimalVals[i] = values.max()
            oneHot = np.zeros(len(moves))
            oneHot[values.argmax()] = 1
            optimalPolicies[i] = oneHot
            states[i] = py222.getState(sample).flatten()
        train(states, optimalVals, optimalPolicies, sess)




