from osim.env import RunEnv
import numpy as np
import time
#env = RunEnv(visualize=False)
env = RunEnv(visualize=False)
observation = env.reset(difficulty=0)
	
episodes  = 100000
for episode in range(0, episodes):

    #env.step(action)
    # action is a list of length 18. values between [0,1]
    ## specifics: 9 muscles per leg, 2 legs = 18. 
    action = env.action_space.sample();
    for i in range(1,1000):
        observation, reward, done, info = env.step(action)
#        env.render()
#        print(observation)
#        time.sleep(1)
        if done:
            env.reset(difficulty = 0, seed = None) #resets the environment if done is true
            print("reseting environment" + str(episodes))
 
