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

class CACLA:

    def GetCritic(self):
        if self.critic is None:
            try:
                model = load_model('Critic.h5')
            except:
                network = Sequential()
                network.add(Dense(64, input_dim=41))
                network.add(Activation('relu'))
                network.add(Dense(15))
                network.add(Activation('relu'))
                network.add(Dense(1))
                network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
                model = network
                model.save("Critic.h5")
        else:
            model = self.critic
        return model

    def GetActor(self):
        if self.actor is None:
            try:
                model = load_model('Actor.h5')
            except:
                network = Sequential()
                network.add(Dense(64, input_dim=41))
                network.add(Activation('relu'))
                network.add(Dense(15))
                network.add(Activation('relu'))
                network.add(Dense(18))
                network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
                model = network
                model.save("Actor.h5")
        else:
            model = self.actor
        return model



    def __init__(self, alpha, gamma):
        self.critic = None
        self.critic = self.GetCritic() 
        self.actor = None
        self.actor = self.GetActor()
        self.state = np.zeros(41)
        self.action = np.zeros(18)
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
    def SaveNetwork(self):
        criticSaved = False
        actorSaved = False
        if not self.critic is None:
            self.critic.save("Critic.h5")
            criticSaved = True
        if not self.actor is None:
            self.actor.save("Actor.h5")
            actorSaved = True 
        return criticSaved and actorSaved 

    def Update(self,ds, da, reward):
        V = self.critic.predict(np.tile(self.state,(2,1)), verbose=0)
        dV = self.critic.predict(np.tile(ds,(2,1)), verbose=0)
        #Actual update
        Vupdate = self.alpha * (reward + self.gamma * dV - V) + V
        self.critic.fit(np.tile(self.state,(2,1)), Vupdate, verbose=0)
        if Vupdate[0] > 0:
            self.actor.fit(np.tile(self.state,(2,1)), np.tile(da, (2,1)), verbose=0)

        self.state = ds
        self.action = da

    def EpsilonPolicy(self, epsilon):
        random.seed(time.time())
        p = random.random() 
        a = np.zeros(18)
        if (p < epsilon):
            random.seed(time.time())
            return np.random.rand(18) 
        else:
            a = self.actor.predict(np.tile(self.state,(2,1)), verbose=0)
            return a[0]
