from NeuralGeneticAlgorithm import NeuralGeneticAlgorithm
import random
import torch
import torch.nn as nn
import numpy as np

def fitness(param,i, d):
    print(i)
    d[i] = np.random.uniform(-1, 1)
    return np.random.uniform(-1, 1)


if __name__ == "__main__":
    alg = NeuralGeneticAlgorithm(fitness, population = 1000, toKeep = 100 )
    alg.generateInitialPopulation(hiddenlayer=100, state=30, action=1)
    for i in range(0,10):
        alg.generateScores()

        alg.selectParents(dev=1);
        alg.performCrossOver(nummix=2)
        alg.performMutation()


