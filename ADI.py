import py222
from random import randint
import numpy as np
import tensorflow as tf
import os
import sys
from scipy.sparse import coo_matrix
import collections
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

kLearningRate = 0.0005
kNumStickers = 24
kNumCubes = 8
kModelPath = "./model.cpkt"
kMCTSExploration = 4.0 #this is form alphago, can't dispute it
kDiscountFactor = 1.0
kMCTSSimulateIterations = 100
kLambda = 1

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

def getFileEnding(i, M):
    numTotalDigits = len(str(M))
    numCurrDigits = len(str(i))
    leadingZeros = '0' * (numTotalDigits - numCurrDigits)
    return "{}{}".format(leadingZeros, i)

def createScrambledCube(numScrambles):
    cube = py222.initState()
    for i in range(numScrambles):
        cube = py222.doAlgStr(cube, getRandomMove())
    return cube

def generateTrainingSet(k, l, M):
    for i in range(M):
        samples, states = generateSamples(k, l)
        fileEnding = getFileEnding(i, M)
        np.save("trueCubes{}".format(fileEnding), samples)
        np.save("states{}".format(fileEnding), states)

# TODO: Add the loss weight to each sample?
def generateSamples(k, l):
    N = k * l
    samples = np.empty((N, kNumStickers), dtype=bytes)
    states = np.empty((N, kNumCubes * kNumStickers))
    for i in range(l):
        currentCube = py222.initState()
        for j in range(k):
            scrambledCube = py222.doAlgStr(currentCube, getRandomMove())
            samples[k * i + j] = scrambledCube
            states[k * i + j] = py222.getState(scrambledCube).flatten()
            currentCube = scrambledCube
    return samples, coo_matrix(states)

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
        samples, _ = generateSamples(k, l)
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
        print(optimalVals)
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
    return False, maxMoves+1

def solveSingleCubeVanillaMCTS(cube, maxMoves, maxDepth):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        bestMove = selectActionVanillaMCTS(cube, maxDepth)
        if bestMove == -1:
            print("something went wrong when selecting best move")
            break
        cube = py222.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves+1

def selectActionVanillaMCTS(state, depth):
    stateStr = str(state)
    q = {}
    counts = {}
    seenStates = set()
    for i in range(kMCTSSimulateIterations):
        simulateVanillaMCTS(state, depth, q, counts, seenStates, stateStr)
    allVals = np.zeros(len(moves))
    for i in range(len(moves)):
        allVals[i] = q[stateStr][moves[i]]
    return allVals.argmax()

def simulateVanillaMCTS(state, depth, q, counts, seenStates, stateStr):
    if depth == 0:
        return 0
    if stateStr not in seenStates:
        q[stateStr] = {}
        counts[stateStr] = {}
        for move in moves:
            q[stateStr][move] = 0
            counts[stateStr][move] = 1
        seenStates.add(stateStr)
        return rolloutVanillaMCTS(state, depth)
    totalStateCounts = 0
    for move in moves:
        totalStateCounts += counts[stateStr][move]
    allQuantities = np.zeros(len(moves))
    for i in range(len(moves)):
        allQuantities[i] = q[stateStr][moves[i]] + kMCTSExploration * math.sqrt(math.log(totalStateCounts)/counts[stateStr][moves[i]])
    bestActionIndex = allQuantities.argmax()
    bestMove = moves[bestActionIndex]
    nextState = py222.doAlgStr(state, bestMove)
    r = reward(nextState)
    newQ = r + kDiscountFactor * simulateVanillaMCTS(nextState, depth - 1, q, counts, seenStates, str(nextState))
    counts[stateStr][bestMove] += 1
    q[stateStr][bestMove] += (newQ - q[stateStr][bestMove])/counts[stateStr][bestMove] 
    return newQ

def rolloutVanillaMCTS(state, depth):
    if depth == 0:
        return 0
    _, policies = forwardPass(state, sess)
    policiesArray = policies.eval()
    actionIndex = selectActionSoftmax(policiesArray)
    nextState = py222.doAlgStr(state, moves[actionIndex])
    r = reward(nextState)
    return r + kDiscountFactor * rolloutVanillaMCTS(nextState, depth - 1)

def selectActionSoftmax(probabilities):
    probabilities = probabilities[0]
    weights = np.zeros(len(probabilities))
    for i in range(len(probabilities)):
        weights[i] = math.exp(kLambda * probabilities[i])
    return weighted_choice(weights)

#this code stolen off stackoverflow (thank you kind stranger)
def weighted_choice(weights):
    totals = np.cumsum(weights)
    norm = totals[-1]
    throw = np.random.rand()*norm
    return np.searchsorted(totals, throw)

def solveSingleCubeFullMCTS(cube, maxMoves, maxDepth):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        value, policies = forwardPass(cube, sess)

        bestMove = selectActionMCTS(cube, maxDepth)
        cube = py222.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves+1
        

def simulateCubeSolvingGreedy(numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeGreedy(scrambledCube, 3 * currentSolveDistance + 1)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def simulateCubeSolvingVanillaMCTS(numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeVanillaMCTS(scrambledCube, 3 * currentSolveDistance + 1, 4)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def simulateCubeSolvingFullMCTS(numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeFullMCTS(scrambledCube, 3 * currentSolveDistance + 1)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("Invalid number of arguments. Must specify model source (-newmodel or -restoremodel) followed by model prefix (can enter 'default' for default prefix) and search strategy (-greedy, -vanillamcts, -fullmcts)")
    else:
        tf.reset_default_graph()
        nnGraph = tf.Graph()
        constructGraph(nnGraph)
        global sess
        sess = tf.Session(graph=nnGraph)
        with nnGraph.as_default():
            with sess.as_default():
                model_prefix = sys.argv[2]
                if model_prefix == "default":
                    model_prefix = kModelPath
                if sys.argv[1].lower() == "-newmodel":
                    doADI(k=5,l=20,M=10,nnGraph=nnGraph)
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, model_prefix)
                    print("Model saved in path: %s" % save_path)
                elif sys.argv[1].lower() == "-restoremodel":
                    saver = tf.train.Saver()
                    saver.restore(sess, model_prefix)
                    print("Model restored from " + model_prefix)
                else:
                    print("Invalid first argument: must be -newmodel or -restoremodel")

                #only simulate cubes upon restoring model for now. can be removed later
                if sys.argv[1].lower() == "-restoremodel":
                    if sys.argv[3].lower() == "-greedy":
                        simulateCubeSolvingGreedy(numCubes=40, maxSolveDistance=4)
                    if sys.argv[3].lower() == "-vanillamcts":
                        simulateCubeSolvingVanillaMCTS(numCubes=5, maxSolveDistance=4)
                    if sys.argv[3].lower() == "-fullmcts":
                        simulateCubeSolvingFullMCTS(numCubes=40, maxSolveDistance=4)




