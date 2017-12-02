import unittest
import torch
from ReplayMemory import ReplayMemory

class UnitTestReplayMemory(unittest.TestCase):


    def testDefaultInit(self):
        RepMem = ReplayMemory()
        self.assertEqual(RepMem.size, 40)

    def testPush(self):
        RepMem = ReplayMemory()
        for i in range(0, 40):
            RepMem.pushSample(torch.mul(torch.ones(41),i), i)
        for i in range(0, 40):
            curr = RepMem.memory[i,:]
            res = torch.sum(curr)
            val = 41 * i
            self.assertEqual(RepMem.reward[i], i)
            self.assertEqual(res, val) 

        RepMem.pushSample(torch.mul(torch.ones(41),41), 41)
        curr = RepMem.memory[0,:]
        res = torch.sum(curr)
        self.assertEqual(res, 41 * 41),
        self.assertEqual(RepMem.reward[0], 41)

        RepMem.pushSample(torch.mul(torch.ones(41),42), 42)
        curr = RepMem.memory[1,:]
        res = torch.sum(curr)
        self.assertEqual(res, 41 * 42)
        self.assertEqual(RepMem.reward[1], 42)
        
    def testPushInxes(self):
        RepMem = ReplayMemory()

        for i in range(0, 200000):
            RepMem.pushSample(torch.ones(41), i)
            self.assertEqual(RepMem.indx, ((i + 1) % RepMem.size))


if __name__ == '__main__':
    unittest.main()
