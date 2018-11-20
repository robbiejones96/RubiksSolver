import py222
from random import randint
import numpy as np
import tensorflow as tf
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

kLearningRate = 0.0005
kModelPath = "./model.cpkt"

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
    return 1 if py222.isSolved(cube, True) else -2

def forwardPass(cube, sess):
    state = py222.getState(cube).flatten()
    inputState = np.array([state])
    policy = sess.run(policyOutput, feed_dict={X: inputState})
    policy = tf.nn.softmax(policy)
    value = sess.run(valueOutput, feed_dict={X: inputState})
    return value, policy

def train(states, optimalVals, optimalPols, sess):
    optimalVals = np.array([optimalVals]).T
    for i in range(10):
        sess.run(optimizer, feed_dict={X : states, optimalPolicies : optimalPols, optimalValues : optimalVals})

def constructGraph(nnGraph):
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

    with nnGraph.as_default():
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
def doADI(k, l, M, nnGraph):
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
                #print value, _
                #print reward(child)
                values[j] = value + reward(child)
            optimalVals[i] = values.max()
            oneHot = np.zeros(len(moves))
            oneHot[values.argmax()] = 1
            optimalPolicies[i] = oneHot
            states[i] = py222.getState(sample).flatten()
        #print states, optimalVals, optimalPolicies
        # print states.shape, optimalVals.shape, optimalPolicies.shape
        # print optimalVals.T.shape
        print optimalVals
        train(states, optimalVals, optimalPolicies, sess)

def solveSingleCubeGreedy(cube, maxMoves):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        _, policies = forwardPass(cube, sess)
        policiesArray = policies.eval()
        bestMove = policiesArray.argmax()
        cube = py222.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves

def solveSingleCubeMCTS(cube, maxMoves):
    pass
        
def createScrambledCube(numScrambles):
    cube = py222.initState()
    for i in range(numScrambles):
        cube = py222.doAlgStr(cube, getRandomMove())
    return cube

def simulateCubeSolving(numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeGreedy(scrambledCube, 3 * currentSolveDistance + 1)
            print(numMoves, numMoves == 3*currentSolveDistance + 1)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)


if __name__ == "__main__":
    tf.reset_default_graph()
    nnGraph = tf.Graph()
    constructGraph(nnGraph)
    global sess
    sess = tf.Session(graph=nnGraph)
    with nnGraph.as_default():
        with sess.as_default():
            if sys.argv[1].lower() == "-newmodel":
                doADI(k=5,l=20,M=10,nnGraph=nnGraph)
                saver = tf.train.Saver()
                save_path = saver.save(sess, kModelPath)
                print("Model saved in path: %s" % save_path)
            elif sys.argv[1].lower() == "-restoremodel":
                saver = tf.train.Saver()
                saver.restore(sess, kModelPath)
                print("Model restored from " + kModelPath)
                simulateCubeSolving(numCubes=20, maxSolveDistance=3)
            else:
                print("Invalid first argument: must be -newmodel or -restoremodel")




