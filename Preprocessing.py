import torch
import numpy as np


class Preprocessing:

    def __init__(self, observation, delta=0.01):
        self.observation = np.concatenate((observation, np.zeros(14)), axis=0) 
        self.delta = delta

    def calcDeriv(self, p, n):
        return (n - p) / self.delta
    
    
    def GetState(self, obs):
        #rot, x, y psn
        pelvis = obs[0:3]
        # velocity rot, x, y
        pelvisV = obs[3:6] 
        # convert to "relative"
        rotAnkle = obs[6:8] - pelvis[0]

        rotKnee = obs[8:10] - pelvis[0]
        rotHip = obs[10:12] - pelvis[0]

        angV = obs[12:18] - pelvisV[0]
        psnCenterMass = obs[18:20] - pelvis[1:3]
        vCenterMass = obs[20:22] - pelvisV[1:3]
        psnOtherParts = obs[22:36]  - np.repeat(pelvis[1:3], 7)
        
        vOtherParts = self.calcDeriv(self.observation[22:36], psnOtherParts)

        psoasStrength = obs[36:38]
        obstacles = obs[38:40] - pelvis[1:3]
        
        self.observation = np.concatenate((
            pelvis,
            pelvisV,
            rotAnkle,
            rotKnee,
            rotHip,
            angV,
            psnCenterMass,
            vCenterMass,
            psnOtherParts,
            vOtherParts,
            psoasStrength,
            obstacles), 0)

        return self.observation 
    def ConvertToTensor(self,s,a,r,sp):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        r = torch.Tensor([r]).float()
        sp = torch.from_numpy(sp).float()
        return s,a,r,sp
                

        

