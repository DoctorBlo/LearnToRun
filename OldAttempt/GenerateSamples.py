from osim.env import RunEnv
import numpy as np



def RunEpisode(iterations):

    print("starting simulation")
    env = RunEnv(visualize=False)
    samples = np.zeros((iterations,18+41))
    rewards = np.zeros((iterations, 1))
    observation = env.reset(difficulty=0);
    for i in range(0,iterations):
        action = env.action_space.sample();
        observation, reward,  done, info = env.step(action)
        observation = np.asarray( observation, dtype=float)
        
        sample = np.concatenate((action,observation), axis=0);
        samples[i] = sample;
        rewards[i] = reward;
        if (done):
            break

    return samples, rewards


def GetAction(indx):
    action = np.zeros(18)
    action[indx] = 1.0
    return action
