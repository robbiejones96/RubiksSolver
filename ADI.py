import py222
from random import randint
import numpy as np
import tensorflow as tf
import os
import sys
from scipy.sparse import coo_matrix
import collections
import math
import gc
from CubeModel import buildModel, compileModel
from tensorflow.train import RMSPropOptimizer
from tensorflow.keras.models import load_model
import MCTS

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
kVirtualLoss = 2

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

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

kNumMinEpochs = 100
kNumMaxEpochs = 1000
kEpsilon = 0.5

def doADI(k, l, M):
    model = buildModel(kNumStickers * kNumCubes)
    compileModel(model, kLearningRate)
    for _ in range(M):
        samples, _ = generateSamples(k, l)
        states = np.empty((len(samples), kNumStickers * kNumCubes))
        optimalVals = np.empty((len(samples), 1))
        optimalPolicies = np.empty(len(samples), dtype=np.int32)  
        for i, sample in enumerate(samples):
            values = np.empty(len(moves))
            for j, move in enumerate(moves):
                child = py222.doAlgStr(sample, move)
                childState = np.array([py222.getState(child).flatten()])
                value, _ = model.predict(childState)
                value = value[0][0]
                values[j] = value + reward(child)
            optimalVals[i] = np.array([values.max()])
            optimalPolicies[i] = values.argmax()
            states[i] = py222.getState(sample).flatten()
        model.fit(states, {"PolicyOutput" : optimalPolicies,
                           "ValueOutput" : optimalVals}, epochs=kNumMaxEpochs,
                           verbose=False, steps_per_epoch=1)
        gc.collect()
    return model

def solveSingleCubeGreedy(model, cube, maxMoves):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        state = np.array([py222.getState(cube).flatten()])
        _, policies = model.predict(state)
        policiesArray = policies[0]
        bestMove = policiesArray.argmax()
        cube = py222.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves+1

def solveSingleCubeVanillaMCTS(model, cube, maxMoves, maxDepth):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        bestMove = selectActionVanillaMCTS(model, cube, maxDepth)
        if bestMove == -1:
            print("something went wrong when selecting best move")
            break
        cube = py222.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves+1

def selectActionVanillaMCTS(model, state, depth):
    stateStr = str(state)
    q = {}
    counts = {}
    seenStates = set()
    for i in range(kMCTSSimulateIterations):
        simulateVanillaMCTS(model, state, depth, q, counts, seenStates, stateStr)
    allVals = np.zeros(len(moves))
    for i in range(len(moves)):
        allVals[i] = q[stateStr][moves[i]]
    return allVals.argmax()

def simulateVanillaMCTS(model, state, depth, q, counts, seenStates, stateStr):
    if depth == 0:
        return 0
    if stateStr not in seenStates:
        q[stateStr] = {}
        counts[stateStr] = {}
        for move in moves:
            q[stateStr][move] = 0
            counts[stateStr][move] = 1
        seenStates.add(stateStr)
        return rolloutVanillaMCTS(model, state, depth)
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
    newQ = r + kDiscountFactor * simulateVanillaMCTS(model, nextState, depth - 1, q, counts, seenStates, str(nextState))
    counts[stateStr][bestMove] += 1
    q[stateStr][bestMove] += (newQ - q[stateStr][bestMove])/counts[stateStr][bestMove] 
    return newQ

def rolloutVanillaMCTS(model, cube, depth):
    if depth == 0:
        return 0
    state = np.array([py222.getState(cube).flatten()])
    _, policies = model.predict(state)
    actionIndex = selectActionSoftmax(policies)
    nextState = py222.doAlgStr(cube, moves[actionIndex])
    r = reward(nextState)
    return r + kDiscountFactor * rolloutVanillaMCTS(model, nextState, depth - 1)

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

def solveSingleCubeFullMCTS(model, cube, maxMoves, maxDepth):
    numMovesTaken = 0
    simulatedPath = []
    simulatedActions = []
    treeStates = set()
    seenStates = set()
    currentCube = cube
    currentCubeStr = str(cube)
    counts = {}
    maxVals = {}
    priorProbabilities = {}
    virtualLosses = {}
    state = np.array([py222.getState(currentCube).flatten()])
    _, probs = model.predict(state)
    probsArray = probs[0]
    initStateVals(currentCubeStr, counts, maxVals, priorProbabilities, virtualLosses, probsArray)
    seenStates.add(currentCubeStr)
    simulatedPath.append(currentCube)
    while numMovesTaken <= maxMoves:
        if py222.isSolved(currentCube, convert=True):
            return True, numMovesTaken, simulatedPath
        if currentCubeStr not in treeStates:
            for move in moves:
                childState = py222.doAlgStr(currentCube, move)
                childStateStr = str(childState)
                if childStateStr not in seenStates:
                    state = np.array([py222.getState(childState).flatten()])
                    _, probs = model.predict(state)
                    probsArray = probs[0]
                    initStateVals(childStateStr, counts, maxVals, priorProbabilities, virtualLosses, probsArray)
                    seenStates.add(childStateStr)
            state = np.array([py222.getState(currentCube).flatten()])
            value, _ = model.predict(state)
            value = value[0][0]
            for i, state in enumerate(simulatedPath):
                if i < len(simulatedActions):
                    stateStr = str(state)
                    maxVals[stateStr][simulatedActions[i]] = max(maxVals[stateStr][simulatedActions[i]], value)
                    counts[stateStr][simulatedActions[i]] += 1
                    virtualLosses[stateStr][simulatedActions[i]] -= kVirtualLoss
            treeStates.add(currentCubeStr)
        else:
            actionVals = np.zeros(len(moves))
            totalStateCounts = 0
            for move in moves:
                totalStateCounts += counts[currentCubeStr][move]
            for i in range(len(moves)):
                currMove = moves[i]
                q = maxVals[currentCubeStr][currMove] - virtualLosses[currentCubeStr][currMove]
                u = kMCTSExploration * priorProbabilities[currentCubeStr][currMove] * math.sqrt(totalStateCounts)/(1+counts[currentCubeStr][currMove])
                actionVals[i] = u + q
            bestMoveIndex = actionVals.argmax()
            bestMove = moves[bestMoveIndex]
            virtualLosses[currentCubeStr][bestMove] += kVirtualLoss
            simulatedActions.append(bestMove)
            currentCube = py222.doAlgStr(currentCube, bestMove)
            currentCubeStr = str(currentCube)
            simulatedPath.append(currentCube)
            numMovesTaken += 1
    return False, maxMoves+1
        

def simulateCubeSolvingGreedy(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = py222.createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeGreedy(model, scrambledCube, 3 * currentSolveDistance + 1)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def initStateVals(stateStr, counts, maxVals, priorProbabilities, virtualLosses, probs):
    counts[stateStr] = {}
    maxVals[stateStr] = {}
    priorProbabilities[stateStr] = {}
    virtualLosses[stateStr] = {}
    for i, move in enumerate(moves):
        counts[stateStr][move] = 0
        maxVals[stateStr][move] = 0
        virtualLosses[stateStr][move] = 0
        priorProbabilities[stateStr][move] = probs[i]       

def simulateCubeSolvingVanillaMCTS(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeVanillaMCTS(model, scrambledCube, 3 * currentSolveDistance + 1, 4)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def simulateCubeSolvingFullMCTS(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeFullMCTS(model, scrambledCube, 3 * currentSolveDistance + 1)
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
        model_prefix = sys.argv[2]
        if model_prefix == "default":
            model_prefix = kModelPath
        if sys.argv[1].lower() == "-newmodel":
            model = doADI(k=5,l=20,M=10)
            model.save("{}.h5".format(model_prefix))
            print("Model saved in path: {}.h5".format(model_prefix))
        elif sys.argv[1].lower() == "-restoremodel":
            model = load_model("{}.h5".format(model_prefix))
            print("Model restored from " + model_prefix)
        else:
            print("Invalid first argument: must be -newmodel or -restoremodel")

        #only simulate cubes upon restoring model for now. can be removed later
        if sys.argv[1].lower() == "-restoremodel":
            if sys.argv[3].lower() == "-greedy":
                simulateCubeSolvingGreedy(model, numCubes=40, maxSolveDistance=4)
            if sys.argv[3].lower() == "-vanillamcts":
                simulateCubeSolvingVanillaMCTS(model, numCubes=5, maxSolveDistance=4)
            if sys.argv[3].lower() == "-fullmcts":
                simulateCubeSolvingFullMCTS(model, numCubes=40, maxSolveDistance=4)




