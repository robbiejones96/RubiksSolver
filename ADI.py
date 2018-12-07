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
import constants
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

# TODO: Add the loss weight to each sample?
def generateSamples(k, l):
    N = k * l
    samples = np.empty((N, constants.kNumStickers), dtype=bytes)
    states = np.empty((N, constants.kNumCubes * constants.kNumStickers))
    for i in range(l):
        currentCube = py222.initState()
        for j in range(k):
            scrambledCube = py222.doAlgStr(currentCube, getRandomMove())
            samples[k * i + j] = scrambledCube
            states[k * i + j] = py222.getState(scrambledCube).flatten()
            currentCube = scrambledCube
    return samples, coo_matrix(states)

def reward(cube):
    return 1 if py222.isSolved(cube, True) else -1

def doADI(k, l, M):
    model = buildModel(constants.kNumStickers * constants.kNumCubes)
    compileModel(model, constants.kLearningRate)
    for iterNum in range(M):
        t0 = time.time()
        samples, _ = generateSamples(k, l)
        t1 = time.time()
        print(t1-t0)
        states = np.empty((len(samples), constants.kNumStickers * constants.kNumCubes))
        optimalVals = np.empty((len(samples), 1))
        optimalPolicies = np.empty(len(samples), dtype=np.int32)  
        t0 = time.time()
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
        t1 = time.time()
        print(t1-t0)
        t0 = time.time()
        model.fit(states, {"PolicyOutput" : optimalPolicies,
                           "ValueOutput" : optimalVals}, epochs=constants.kNumMaxEpochs,
                           verbose=False, steps_per_epoch=1)
        t1 = time.time()
        print(t1-t0)
        gc.collect()
        print(iterNum)
    return model


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("Invalid number of arguments. Must specify model source (-newmodel or -restoremodel) followed by model prefix (can enter 'default' for default prefix) and search strategy (-greedy, -vanillamcts, -fullmcts)")
    else:
        model_prefix = sys.argv[2]
        if model_prefix == "default":
            model_prefix = constants.kModelPath
        if sys.argv[1].lower() == "-newmodel":
            model = doADI(k=20,l=1,M=100)
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
                MCTS.simulateCubeSolvingGreedy(model, numCubes=50, maxSolveDistance=7)
            if sys.argv[3].lower() == "-vanillamcts":
                MCTS.simulateCubeSolvingVanillaMCTS(model, numCubes=50, maxSolveDistance=7)
            if sys.argv[3].lower() == "-fullmcts":
                MCTS.simulateCubeSolvingFullMCTS(model, numCubes=50, maxSolveDistance=7)




