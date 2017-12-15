from osim.env import RunEnv
import numpy as np
import time
from Preprocessing import Preprocessing

import sys
#sys.path.insert(0, sys.path[0] + '/DDPG/DDPG.py')

from DDPG.DDPG import DDPG

#env = RunEnv(visualize=False)
env = RunEnv(visualize=True)
observation = env.reset(difficulty=0)

episodes  = 100000
agent = DDPG(.9, 2000, 54, 18, .0001,criticpath='critic', actorpath='actor')

for episode in range(0, episodes):

    #env.step(action)
    # action is a list of length 18. values between [0,1]
    ## specifics: 9 muscles per leg, 2 legs = 18.
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    observation = np.array(observation)
    Preprocess = Preprocessing(observation, delta=0.01)
    prevState = Preprocess.GetState(observation)
    agent.step = 0
    agent.OUprocess(.312, 0.15, 0.0)
    for i in range(1,1000):
        if i > 1:
            pelvis_y = observation[2]
        observation, reward, done, info = env.step(action)

        if i > 1:
            reward += (observation[2] - pelvis_y)*0.01 #penalty for pelvis going down
        reward = env.current_state[4] * 0.01
        reward += 0.01  # small reward for still standing
        reward += min(0, env.current_state[22] - env.current_state[1]) * 0.1  # penalty for head behind pelvis
        reward -= sum([max(0.0, k - 0.1) for k in [env.current_state[7], env.current_state[10]]]) * 0.02  # penalty for straight legs

        observation = np.array(observation)
        #means it didn't go the full simulation
        if done:
            reward = 0
        state = Preprocess.GetState(observation)
        s,a,r,sp = Preprocess.ConvertToTensor(prevState,action, reward, state)
        agent.addToMemory(s,a,r,sp)

        if(agent.primedToLearn()):
            agent.PerformUpdate(16)
            agent.UpdateTargetNetworks()
#        env.render()
        if done:
            env.reset(difficulty = 0, seed = None) #resets the environment if done is true
            agent.saveActorCritic()
            print("reseting environment" + str(episode))
            break
        action = agent.selectAction(s)
        action = action.numpy()
        prevState = state;
