import py222
from random import randint, getrandbits


moves = ['F', 'B', 'R', 'L', 'D', 'U']

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
