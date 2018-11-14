import py222
from random import randint
import numpy as np

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

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

def forwardPass(cube):
    # TODO: run cube through neural net
    return 0, np.empty(12) # TODO: REMOVE THIS

def train(states, optimalVals, optimalPolicies):
    pass

# TODO: might need to change numpy arrays to TF variables
def doADI(k, l, M):
    for _ in range(M):
        samples = generateSamples(k, l)
        states = np.empty((len(samples), 8 * 24))
        optimalVals = np.empty(len(samples))
        optimalPolicies = np.empty((len(samples), len(moves)))
        for i, sample in enumerate(samples):
            values = np.empty(len(moves))
            for j, move in enumerate(moves):
                child = py222.doAlgStr(sample, move)
                value, policy = forwardPass(child)
                values[j] = value + reward(child)
            optimalVals[i] = values.max()
            oneHot = np.zeros(len(moves))
            oneHot[values.argmax()] = 1
            optimalPolicies[i] = oneHot
            states[i] = py222.getState(sample).flatten()
        train(states, optimalVals, optimalPolicies)




