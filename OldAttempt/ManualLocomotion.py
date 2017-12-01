import numpy as np



def FallingPhase():
    action = np.zeros(18) 
    action[8] = .5
    action[17] = .5 
    action[2] = .25
    action[3] = .25
    action[4] = .5
    action[1] = .25
    action[10] = .25
    #action[5] = 1.0
    action[12] = .25
    action[11] = .25
    action[13] = .5
    #action[14] = 1.0
    return action
def RunningMotion(i):
    a = np.zeros(18) + 0 
    ang = i* np.pi / 100 
    u = np.cos(ang)
    d =  np.cos(ang + np.pi)
    print("leg cycle")
    print(u)
    print(d)
    if (d > 0):
        a[0] = 0.1
        a[1] = 0.1 
        a[2] =1 
        a[3] = 0.5
        a[4] = 0.4
        a[5] = 0.4
        a[6] = .2
        a[7] = .2
        a[8] = .4
        #left leg
        a[9] = .5 
        a[10] = 1.0
        a[11] = 0.2 
        a[12] = 1.0
        a[13] = .5
        a[14] = 0
        a[15] = .2
        a[16] = .2
        a[17] = .4
    else:
        a[0] = .5 #Right hamstring
        a[1] = 1.0 #right bicep femoris
        a[2] = 0.2 # Gluteous maximums
        a[3] = 1.0 # Right illios psoas
        a[4] = 0.5  # Right rectus femorus (quad)
        a[5] = 0 # right vastus (quad)
        a[6] = .2# right back calv
        a[7] = .2 # right soleus lower back calve
        a[8] = .4 # tibialis anterior right
        #left leg
        a[9] = 0.1
        a[10] = .1
        a[11] = 1.0
        a[12] = 0.5
        a[13] = .4
        a[14] = .4
        a[15] = .2
        a[16] = .2
        a[17] = .4
        
    return a 
def RaiseRightLeg():
    a = np.zeros(18)
    a[3] = 1.0
    a[4] = .25
    a[5] = .25
    a[6] = .25
    a[7] = .25
    a[8] = 1.0
    a[9] = 0.7
    a[10] = 0.7
    a[11] = 1.0
    a[12] = .5
    a[13] = .25
    a[14] = .25
    a[15] = .5
    a[16] = .5
    action = a
    return a


def RaiseLeftLeg():
    a = np.zeros(18)
    a[12] = 1.0
    a[13] = .5
    a[14] = .5
    a[15] = .5
    a[16] = .5
    a[17] = 1.0
    a[0] = 0.7
    a[1] = 0.7
    a[2] = 1.0
    a[3] = .5
    a[4] = .25
    a[5] = .25
    a[6] = .5
    a[7] = .5
    action = a
    return a

