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
        target.OUprocess(random.random(), 0.15,0.0)
        pelvis_y = 0

        for i in range(1,1000):
            observation, reward, done, info = env.step(action)
            observation = np.array(observation)
            #means it didn't go the full simulation
            if i > 1:
                reward += (observation[2] - pelvis_y)*0.01 #penalty for pelvis going down
            reward = env.current_state[4] * 0.01
            reward += 0.01  # small reward for still standing
            reward += min(0, env.current_state[22] - env.current_state[1]) * 0.1  # penalty for head behind pelvis
            reward -= sum([max(0.0, k - 0.1) for k in [env.current_state[7], env.current_state[10]]]) * 0.02  # penalty for straight legs


            if done and i < 1000:
                reward = 0

            state = Preprocess.GetState(observation)
            s,a,r,sp = Preprocess.ConvertToTensor(prevState,action, reward, state)
            target.addToMemory(s,a,r,sp)

                #        env.render()
            if done:
                env.reset(difficulty = 0, seed = None) #resets the environment if done is true
                if(target.primedToLearn()):

                    lock.acquire()
                    proxy_agent.PerformUpdate(batchsize, target)
                    target.UpdateTargetNetworks(agent.getCritic(), agent.getActor())
                    print("saving actor")
                    proxy_agent.saveActorCritic()
                    print("actor saved")
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
    for i in range(0, 20):
        pool.apply_async(func=Simulation, args=(agent,10))#1000))
    pool.close()
    pool.join()
    print("cool beans")
