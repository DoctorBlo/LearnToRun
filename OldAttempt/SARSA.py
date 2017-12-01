from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.models import load_model
import numpy as np
import time
import random

"""
We going to try something "Simple"
Using a Neural Network as my function approximator I'm going to discretize the state space so
we ever only fire 1 muscle at a time. This has the obvious limitation of not utilizing the full
state,action space and could be missing out on an optimal form of locomotion. 

That being said, I want to start prototyping some work, and using what I know is probably a good place
and should at least give some baseline on how to proceed and what to learn.

The idea is then then This
Create class SARSANN which contains:
    members:
        a Neural Network using Keras
        state - last state
        action - last action
    Functions to load network/ save network
    Update function: updates the Q value by SARSA update policy
    Choose next action basd on current state

    my states: are the observed state + previous action technically
"""

class SARSA:

    def GetNetwork(self):
        if self.network is None:
            try:
                model = load_model('SARSA.h5')
            except:
                network = Sequential()
                network.add(Dense(64, input_dim=59))
                network.add(Activation('relu'))
                network.add(Dense(15))
                network.add(Activation('relu'))
                network.add(Dense(1))
                network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
                model = network
                model.save("SARSA.h5")
        else:
            model = self.network
        return model



    def __init__(self, alpha, gamma):
        self.network = None
        self.network = self.GetNetwork() 
        self.input = np.zeros(59)
        self.state = np.zeros(41)
        self.action = np.zeros(18)
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
    def SaveNetwork(self):
        if not self.network is None:
            self.network.save("SARSA.h5")
            return True
        else:
            return False
    def BatchUpdate(self, states, rewards): 
        Q = self.network.predict(np.tile(self.input, (3,1)), batch_size = 1, verbose =0)
        Q = Q[0]
        dQ = self.network.predict(states,  batch_size = 10, verbose = 0) 

        Qupdate = self.alpha* ( rewards + self.gamma * dQ - Q) + Q
        self.network.fit(np.tile(self.input, (rewards.size,1)), Qupdate, verbose =0);

    def Update(self,ds, da, reward):
        Q = self.network.predict(np.tile(self.input,(2,1)), verbose=0)
        i = np.concatenate((ds,da))
        dQ = self.network.predict(np.tile(i,(2,1)), verbose=0)
        #Actual update
        Qupdate = self.alpha * (reward + self.gamma * dQ - Q) + Q
        self.network.fit(np.tile(self.input,(2,1)), Qupdate, verbose=0)
        self.state = ds
        self.action = da
        self.input = i
    def EpsilonPolicy(self, epsilon):
        random.seed(time.time())
        p = random.random() 
        a = np.zeros(18)
        if (p < epsilon):
            random.seed(time.time())
            toSet = random.randint(0,17)
            a[toSet] = 1.0
            return a 
        else:
            a[self.GenerateBestAction()] = 1.0;
            return a
    def GenerateBestAction(self):
        Qvalues = np.zeros((18, 59))
        for i in range(0, 17):
            a = np.zeros(18);
            a[i] = 1.0
            Qvalues[i,:] = np.concatenate((self.state, a));
        Qvalues = self.network.predict(Qvalues, verbose = 0)
        return np.argmax(Qvalues)
