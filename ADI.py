import py222
from random import randint
import numpy as np

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']


def generateSolvedCubes():
    solvedCubes = [None] * 7
    solvedCubes[0] = py222.initState()
    solvedCubes[1] = py222.doAlgStr(solvedCubes[0], 'y')
    solvedCubes[2] = py222.doAlgStr(solvedCubes[1], 'y')
    solvedCubes[3] = py222.doAlgStr(solvedCubes[2], 'y')
    solvedCubes[4] = py222.doAlgStr(solvedCubes[0], 'x')
    solvedCubes[5] = py222.doAlgStr(solvedCubes[4], 'x')
    solvedCubes[6] = py222.doAlgStr(solvedCubes[5], 'x')
    return solvedCubes

solvedCubes = generateSolvedCubes()

def isSolved(cube):
    for solvedCube in solvedCubes:
        if np.array_equal(cube, solvedCube):
            return True
    return False

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
        
def getChildren(cube):
    children = [None] * len(moves)
    for i in range(len(moves)):
        children[i] = py222.doAlgStr(cube, moves[i])
    return children

def reward(cube):
    return 1 if isSolved(cube) else -1

def forwardPass(cube):
    # TODO: run cube through neural net

    return 0, np.empty(12) # TODO: REMOVE THIS

def train(samples , optimalVals, optimalPolicies):
    pass

def doADI(k, l, M):
    for _ in range(M):
        samples = generateSamples(k, l)
        optimalVals = np.empty(len(samples))
        optimalPolicies = np.empty((len(samples), len(moves)))
        for i, sample in enumerate(samples):
            children = getChildren(sample)
            values = np.empty(len(children))
            policies = np.empty((len(children), len(moves)))
            for j, child in enumerate(children):
                value, policy = forwardPass(child)
                values[j] = value
                policies[j] = policy
            optimalVals[i] = values.max()
            optimalPolicies[i] = policies[values.argmax()]
        train(samples, optimalVals, optimalPolicies)




