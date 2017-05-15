import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random

N = 100
blocks = 600
maxiters = 1000000

# A is a matrix to quickly find all neighbours
a = np.diag([1 for n in np.arange(N)],1)
A = np.matrix(a.T+a)
# B is the boundary matrix, relating x=0 and x=N
B = np.zeros((N+1,N+1))
B[N,0], B[0,N] = 1, 1
B = np.matrix(B)

def newSeed():
    # New seed is placed on top of sink in the middle
    O = np.zeros((N+1,N+1))
    for i in np.arange(5):
        O[int(N/2.0)+i,1] = 1
    return np.matrix(O)

def getGrowthSites(O):
    # Get growth sites
    S = A*O + O*A + B*O
    S[S>0] = 1
    S -= O
    return S

def doRandomWalk(pos, DLA, S):
    steps = [
        np.array([-1,0]), np.array([1,0]), np.array([0,-1]), np.array([0,1])
        ]

    i = 0
    while i < maxiters:
        i += 1
        newpos = pos + random.choice(steps)
        newpos[0] = newpos[0]%(N+1) # apply boundary conditions
        x, y = newpos
        # The random walker is not allowed to leave at max of min y values
        if y == 0 or y == N+1:
            break
        # The random walker is not allowed to walk through existing sites
        if DLA[x,y] == 1:
            continue
        pos = newpos
        # If a growth site is reached, there is a change of sticking
        if S[x,y] == 1 and np.random.random() < p:
            DLA[x,y] = 1
            return DLA, True

    return DLA, False

if __name__ == '__main__':
    for p in [0.1]:
        DLA = newSeed()

        for b in np.arange(blocks):
            if b%50 == 0:
                print("Block ",b)
            S = getGrowthSites(DLA)
            placed = False

            while not placed:
                x = random.choice(np.arange(N))
                init = np.array([x,N])
                DLA, placed = doRandomWalk(init, DLA, S)

        plt.matshow(DLA)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.show()
