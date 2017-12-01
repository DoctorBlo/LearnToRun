from osim.env import RunEnv
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Input, concatenate
import numpy as np

from GenerateSamples import RunEpisode
from GenerateSamples import GetAction

from SARSA import SARSA
from CACLA import CACLA

from ManualLocomotion import FallingPhase, RaiseRightLeg, RaiseLeftLeg, RunningMotion 
cacla = CACLA(.1,.9)
"""
for episodes in range(1):
    samples, rewards = RunEpisode(100)
    sarsa.BatchUpdate(samples, rewards)
    #sarsa.network.fit(samples, rewards, epochs = 10, batch_size=32)
sarsa.SaveNetwork()
"""   
#env = RunEnv(visualize=False)
env = RunEnv(visualize=True)
observation = env.reset(difficulty=0)
runs = 100000
for episodes in range(50000, runs):

    #env.step(action)
    # action is a list of length 18. values between [0,1]
    ## specifics: 9 muscles per leg, 2 legs = 18. 
    action = env.action_space.sample(); 
    for i in range(1,1000):
        observation, reward, done, info = env.step(action)
        if(reward > 1.0 ):
            #only give reward if we are seeing big improvements
            reward = .1
        elif( reward < -1.0):
            reward = -.1
        else:
            reward = 0.0
        cacla.Update(observation, action, reward)
        
        #action = cacla.EpsilonPolicy(1 - episodes / runs)
        action = cacla.EpsilonPolicy(0.1)
        if done:
            if i < 980:
                reward = -0.01
            else:
                reward = 100
            cacla.Update(observation, action, reward)
            cacla.SaveNetwork()
            env.reset(difficulty = 0, seed = None) #resets the environment if done is true
            print("reseting environment" + str(episodes))
            break

