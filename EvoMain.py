from osim.env import RunEnv
import numpy as np
import time
import random
import multiprocessing
import time
from multiprocessing.managers import BaseManager
import copy_reg, copy, pickle
from torch.autograd import Variable

from Preprocessing import Preprocessing
from EvolutionaryLearning.NeuralGeneticAlgorithm import NeuralGeneticAlgorithm

def Simulation(proxy_agent,index, return_dict,  episodes):
    print('starting simulation')
    env = RunEnv(visualize=False)
    observation = env.reset(difficulty=0)

    rewards = np.zeros(episodes)
    totalreward = 0
    for episode in range(0, episodes):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation = np.array(observation)
        Preprocess = Preprocessing(observation, delta=0.01)
        prevState = Preprocess.GetState(observation)
        for i in range(1,1000):
            observation, reward, done, info = env.step(action)
            observation = np.array(observation)
            #means it didn't go the full simulation
            if done and i < 1000:
                reward = 0  

            state = Preprocess.GetState(observation)
            s,a,r,sp = Preprocess.ConvertToTensor(prevState,action, reward, state)

            totalreward += reward
            if done:
                env.reset(difficulty = 0, seed = None) #resets the environment if done is true
                print("reseting environment" + str(episode))
                rewards[episode] = totalreward
                totalreward = 0
                break
            action = proxy_agent(Variable(s, volatile=True))
            action = action.data.numpy()
            prevState = state;
    return_dict[index] = np.sum(rewards) / episodes
    return np.sum(rewards) / episodes

def init(l):
    global lock
    lock = l

class MyManager(BaseManager): pass

def Manager():
   m = MyManager()
   m.start()
   return m

MyManager.register('NeuralGeneticAlgorithm', NeuralGeneticAlgorithm)

def pickle_NN(nn):
    print("Pickling a nn")
    return NeuralGeneticAlgorithm, (NeuralGeneticAlgorithm.generateScore,)
def update(proxy_agent, index, fitness):
    print("we are in update")
    proxy_agent.generateScore(fitness, index)

if __name__ == '__main__':
    manager = Manager()
    l = multiprocessing.Lock()
    episodes = 1
    pop = 100
    sim = lambda a, i, d: Simulation(a,i, d,  episodes)
    NeuroNN = NeuralGeneticAlgorithm(sim, population=pop, mutation=1e-1, toKeep=10)
    NeuroNN.generateInitialPopulation( hiddenlayer=150,state=54,action=18)
    for i in range(0, 100):
        pool = multiprocessing.Pool(initializer=init, initargs=(l,))
        if (i == 0):
            print 'Number of Processes: ', pool._processes
        NeuroNN.generateScores()

        NeuroNN.selectParents(dev=3);
        print("Current Mean Performance: " + str(NeuroNN.getMeanPerformance()))
        NeuroNN.performCrossOver()
        NeuroNN.performMutation()
        NeuroNN.savePopulation()

    print("cool beans")
