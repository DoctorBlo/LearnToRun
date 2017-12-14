import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
import copy
import random
import time
from scipy import stats
import multiprocessing
from multiprocessing.managers import BaseManager
import time

class NeuralGeneticAlgorithm:

    def __init__(self,scorefunction,  population=int(1e3), mutation=1e-1, toKeep=100, savedir='population'):
        self.size = population
        self.mutation = mutation
        self.toKeep = toKeep
        self.population = [None] * int(population)
        self.parents = [] 
        self.fitness = scorefunction
        self.hiddenlayer = 0
        self.state = 0
        self.action = 0
        self.mean = 0 #Generations mean score
        self.dir = savedir
    def generateInitialPopulation(self, hiddenlayer=150,state=41,action=18):
        self.hiddenlayer = hiddenlayer
        self.state = state
        self.action = action
        for i in range(0, self.size):
            net = nn.Sequential(
                    nn.Linear(state, hiddenlayer ),
                    nn.ELU(),
                    nn.Linear(hiddenlayer, action),
                    nn.Sigmoid())
            net = self.initializeNetworkWeights(net)
            score = 0
            self.population[i] = {'net': net, 'score': score}
    def initializeNetworkWeights(self, net):
        for param in net.parameters():
            
            if(len(param.data.size()) < 2):
                continue
            torch.nn.init.xavier_uniform(param.data, gain=np.random.randint(1, 10))

        return net

    def selectParents(self, dev=3):
        scores = np.zeros(self.size)
        for i in range(0, self.size):
            scores[i] = self.population[i]['score']
        self.mean = np.mean(scores) 
        scores = stats.zscore(scores)
        strongest = np.sum(scores >= dev)
        if(strongest == 0):
            print("no strong parents")
        self.parents = [None] * min(self.toKeep, strongest)

        indx = 0
        for i in range(0, self.size):
            if(scores[i] >= dev and (indx < self.toKeep) ):
                self.parents[indx] = self.population[i]
                indx += 1 
        if (len(self.parents) < self.toKeep):
            additional = random.sample(self.population, self.toKeep - len(self.parents)) 
            self.parents = self.parents + additional



    def performCrossOver(self, nummix=2):
        nummix = 2 #future work, allow more than 2 parent
        pairs = np.random.permutation(len(self.parents))
        children = [None] * len(self.parents)
        
        for i in range(0, len(self.parents) - 1, nummix):
            par1 = self.parents[pairs[i]] 
            par2 = self.parents[pairs[i + 1]] 
            res1, res2 = self.mixGenes(par1, par2)
            children[i] = res1
            children[i + 1] = res2
        newgeneration = self.parents + children
        toreplace = np.random.permutation(self.size)
        for i in range(0, len(newgeneration)): 
            self.population[toreplace[i]] = newgeneration[i]
        

    def mixGenes(self, par1, par2):
        res1 = copy.deepcopy(par1['net'])
        res2 = copy.deepcopy(par2['net'])
        toMix = np.random.permutation(self.hiddenlayer)
        res1.state_dict()['0.weight'][toMix,:] = par2['net'].state_dict()['0.weight'][toMix,:]
        res2.state_dict()['0.weight'][toMix,:] = par1['net'].state_dict()['0.weight'][toMix,:]
        res1 = {'net': res1, 'score': 0}
        res2 = {'net': res2, 'score': 0}
        return res1,res2

    def performMutation(self):
        for i in range(0, self.size):
            if(self.mutation > random.random()):
                curDict = self.population[i]['net']
                noise = np.random.uniform(-1, 1, (self.hiddenlayer, self.state))
                curDict.state_dict()['0.weight'] += torch.from_numpy(noise).float()
                self.population[i]['net'] = curDict
             
    def generateScores(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        scores = [multiprocessing.Process(target=self.fitness, \
                args=(self.population[i]['net'], i,  return_dict)) \
                for i in range(0,self.size)]

        for i in range(0,self.size):
            scores[i].start();
        for i in range(0,self.size):
            scores[i].join()

        for i in range(0,self.size):
            print(return_dict[i])
            self.population[i]['score'] = return_dict[i]
        
    def getMeanPerformance(self):
        return self.mean

    def savePopulation(self):
        for i in range(0,self.size):
            model = self.population[i]['net']
            torch.save(model.state_dict(), './' +  self.dir + '/' + str(i) + 'net')


        
