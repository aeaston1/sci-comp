import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas # for printing matrices
import copy

# Set up parameters
N = 3
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
delta_xy = 1/float(N)
maxiters = 2000
growthSteps = 6001
epsilon = 0.005 # Break-off value for convergence
omega = 1.5 # For SOR algoritm

def matrix():
    # A is a matrix to quickly add all neighbours
    a = np.diag([-4 for n in np.arange((N+1)**2)], 0)
    b = np.diag([1 for n in np.arange((N+1)**2-1)], 1)
    c = np.diag([1 for n in np.arange((N+1)**2-N-1)], N+1)
    M = np.matrix(c.T+b.T+a+b+c)
    return M

if __name__ == "__main__":
    M = matrix()
    print(M)
