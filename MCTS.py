import py222
from random import randint
import numpy as np
import tensorflow as tf
import os
import sys
from scipy.sparse import coo_matrix
import collections
import math

def forwardPass(cube, sess):
    state = py222.getState(cube).flatten()
    inputState = np.array([state])
    policy = sess.run(policyOutput, feed_dict={X: inputState})
    policy = tf.nn.softmax(policy)
    value = sess.run(valueOutput, feed_dict={X: inputState})
    return value, policy

def solveSingleCubeGreedy(cube, maxMoves, sess):
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

def solveSingleCubeVanillaMCTS(cube, maxMoves, maxDepth, sess):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        bestMove = selectActionVanillaMCTS(cube, maxDepth, sess)
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
        simulateVanillaMCTS(state, depth, q, counts, seenStates, stateStr, sess)
    allVals = np.zeros(len(moves))
    for i in range(len(moves)):
        allVals[i] = q[stateStr][moves[i]]
    return allVals.argmax()

def simulateVanillaMCTS(state, depth, q, counts, seenStates, stateStr, sess):
    if depth == 0:
        return 0
    if stateStr not in seenStates:
        q[stateStr] = {}
        counts[stateStr] = {}
        for move in moves:
            q[stateStr][move] = 0
            counts[stateStr][move] = 1
        seenStates.add(stateStr)
        return rolloutVanillaMCTS(state, depth, sess)
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

def rolloutVanillaMCTS(state, depth, sess):
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

def solveSingleCubeFullMCTS(cube, maxMoves, maxDepth, sess):
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py222.isSolved(cube, convert=True):
            return True, numMovesTaken
        value, policies = forwardPass(cube, sess)

        bestMove = selectActionMCTS(cube, maxDepth)
        cube = py222.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves+1
        

def simulateCubeSolvingGreedy(numCubes, maxSolveDistance, sess):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = py222.createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeGreedy(scrambledCube, 3 * currentSolveDistance + 1, sess)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def simulateCubeSolvingVanillaMCTS(numCubes, maxSolveDistance, sess):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = py222.createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeVanillaMCTS(scrambledCube, 3 * currentSolveDistance + 1, 4, sess)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)

def simulateCubeSolvingFullMCTS(numCubes, maxSolveDistance, sess):
    data = np.zeros(maxSolveDistance+1)
    for currentSolveDistance in range(maxSolveDistance+1):
        numSolved = 0
        for j in range(numCubes):
            scrambledCube = py222.createScrambledCube(currentSolveDistance)
            result, numMoves = solveSingleCubeFullMCTS(scrambledCube, 3 * currentSolveDistance + 1, sess)
            print(numMoves, numMoves != 3*currentSolveDistance + 2)
            if result:
                numSolved += 1
        percentageSolved = float(numSolved)/numCubes
        data[currentSolveDistance] = percentageSolved
    print(data)
