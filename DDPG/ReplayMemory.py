import torch 

class ReplayMemory:
    def __init__(self, size = 40, dims = 41):
        self.memory = torch.zeros(size, dims)
        self.reward = torch.zeros(size)
        self.indx = 0
        self.size = size 

    def pushSample(self, sample, reward):
        self.memory[self.indx, : ] = sample
        self.reward[self.indx] = reward
        self.indx = (self.indx + 1) % self.size

    def batchSample(self, batchSize):
        shuffled = torch.randperm(self.size)
        ret = shuffled[0:batchSize]
        return (self.memory[toret,:], self.reward[ret,:])

    def singleSample(self):
        shuffled = torch.randperm(self.size)
        return ( self.memory[shuffled[0], :], self.reward[[shuffled[0]]])
