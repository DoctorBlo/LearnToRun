from osim.env import RunEnv
import numpy as np
import time
import random
import multiprocessing
from multiprocessing.managers import BaseManager

from Preprocessing import Preprocessing
from DDPG.AsyncDDPG import AsyncDDPG
from DDPG.TargetActorCritic import TargetActorCritic

#agent = AsyncDDPG(.9, 54, 18, 1e-3)#criticpath='critic', actorpath='actor')

class MyManager(BaseManager): pass

def Manager():
   m = MyManager()
   m.start()
   return m

MyManager.register('AsyncDDPG', AsyncDDPG)

def init(l):
    global lock
    lock = l

	
def Simulation(proxy_agent, episodes):
    env = RunEnv(visualize=False)
    observation = env.reset(difficulty=0)
    memory = random.randint(1000, 2000)
    tau = random.uniform(0.01, .9)
    epsilon = random.uniform(.15, .9)
    target = proxy_agent.ProduceTargetActorCritic( memory, tau, epsilon )	
    batches =  [ 16, 32, 64, 128]
    batchsize = batches[random.randint(0,len(batches)-1)] 

    for episode in range(0, episodes):
        

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation = np.array(observation)
        Preprocess = Preprocessing(observation, delta=0.01)
        prevState = Preprocess.GetState(observation)
        for i in range(1,1000):
            observation, reward, done, info = env.step(action)
            observation = np.array(observation)
            state = Preprocess.GetState(observation)
            s,a,r,sp = Preprocess.ConvertToTensor(prevState,action, reward, state)
            target.addToMemory(s,a,r,sp)

                #        env.render()
            if done:
                env.reset(difficulty = 0, seed = None) #resets the environment if done is true
                if(target.primedToLearn()):
                    lock.acquire()
                    proxy_Agent.PerformUpdate(self, batchsize, target)
                    target.UpdateTargetNetworks(agent.getCritic(), agent.getActor())
                    proxy_Agent.saveActorCritic()
                    lock.release()
                print("reseting environment" + str(episode))
                break
            action = target.selectAction(s)
            action = action.numpy()
            prevState = state;



if __name__ == '__main__':
    manager = Manager()
    agent = manager.AsyncDDPG(.9, 54, 18, 1e-3)
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=init, initargs=(l,))
    print 'Number of Processes: ', pool._processes

    for i in range(0, 10):
        print("hello")
        pool.apply_async(func=Simulation, args=(agent,1000))
    pool.close()
    pool.join()
    print("cool beans")
