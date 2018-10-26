import py222
from random import randint, getrandbits
import numpy as np

moves = ['F', 'B', 'R', 'L', 'D', 'U']


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
    move = moves[randint(0, 5)]
    return move + '\'' if getrandbits(1) else move

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
    children = [None] * len(moves) * 2
    for i in range(len(moves)):
        children[i] = py222.doAlgStr(cube, moves[i])
    for i in range(len(moves)):
        children[i + len(moves)] = py222.doAlgStr(cube, moves[i] + '\'')
    return children

