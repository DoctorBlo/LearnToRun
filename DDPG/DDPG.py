from Actor import actor
from Critic import Critic
from ReplayMemory import ReplayMemory


class DDPG:
    def __init__(self, gamma, memory, state):
        self.memory = ReplayMemory(memory)
        self.actor = Actor()
        self.critic = Critic()
        self.state = torch.zeros(state)
        self.action = torch.zeros(18)

    def selectAction(self):
        return self.actor(self.state)
    def addtoMemory(self, state, reward):
        self.memory.pushSample(state,reward)

    def PerformUpdate(self):
        #Mildly important, according to https://github.com/vy007vikas/PyTorch-ActorCriticRL
        # the criterion on the actor is this: sum(-Q(s,a)) I'm assuming this is over the batch....
        """
        Generate batch from memory
        perform forward pass with actor
        perform forward pass with Critic

        back-prop loss from critic
        back-prop loss form actor

        """

