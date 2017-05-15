import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas # for printing matrices
import copy

# Set up parameters
N = 255 # N must be uneven for checkboard parrallelization
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
delta_xy = 1/float(N)
maxiters = 2000
growthSteps = 6001
epsilon = 0.005 # Break-off value for convergence
omega = 1.5 # For SOR algoritm

# A is a matrix to quickly add all neighbours
a = np.diag([1 for n in np.arange(N)],1)
A = np.matrix(a.T+a)
# B is the boundary matrix, relating x=0 and x=N
B = np.zeros((N+1,N+1))
B[N,0], B[0,N] = 1, 1
B = np.matrix(B)

def analytic(y):
    return y*delta_xy

def newMatrix(O):
    # Build empty matrix
    M = np.zeros((N+1,N+1))
    for y in np.arange(N+1):
        M[:,y] = analytic(y)
    M = setSinksSources(M, O)
    return np.matrix(M)

def checkerboard(N):
    # Code from http://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    N = int(N/2)
    re = np.r_[N*[0, 1]]              # even-numbered rows
    ro = np.r_[N*[1, 0]]              # odd-numbered rows
    return np.row_stack(N*(re, ro))

def setSinksSources(M, O):
    # Set growing object as sinks
    M = np.multiply(M,1 - O)
    # Reset original sinks and sources
    M[:,-1] = 1
    M[:,0] = 0
    return M

def newSeed():
    # New seed is placed on top of sink in the middle
    O = np.zeros((N+1,N+1))
    O[int(N/2.0),1] = 1
    return np.matrix(O)

def SOR(M, O, epsilon, omega):
    k  = 0
    while (k < maxiters):
        k += 1

        # Execute iteration step and reset sinks and sources
        M1 = np.multiply(M, C1)
        M2 = np.multiply(M, C2)
        S1 = (omega/4)*(A*M1 + M1*A + B*M1) + (1-omega)*M2
        S2 = (omega/4)*(A*S1 + S1*A + B*S1) + (1-omega)*M1
        newM = setSinksSources(S1 + S2, O)

        # Check is algorithm is converged
        D = np.abs(newM - M)
        D[np.abs(D) < epsilon] = 0
        if np.sum(D) == 0:
            # print("SOR terminated in iteration ",k) #add timer?
            break

        # Prepare for next iteration
        M = copy.deepcopy(newM)

    return newM, k

def doGrowthStep(M, O):
    # Get growth sites
    S = A*O + O*A + B*O
    S[S>0] = 1
    S -= O

    # Get matrix with growth probability at all sites
    N = np.multiply(S,M)
    N = np.power(N, eden)
    N /= np.sum(N)

    for x, y in coordinates:
        if N[x,y] != 0:
            if np.random.random() < N[x,y]:
                O[x,y] = 1
                M[x,y] = 0

    return M, O

if __name__ == "__main__":
    # Question K
    edens = [1.2]
    C1 = checkerboard(N+1)
    C2 = 1 - C1

    for eden in edens:
        O = newSeed()
        M = newMatrix(O)
        for g in np.arange(growthSteps):
            if g%200 == 0:
                print('Growthstep ',g,' for parameter ',eden)
            M, k = SOR(M, O, epsilon, omega)
            M, O = doGrowthStep(M, O)
        q = 1*np.max(M)
        P = M + q*O
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(P)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        fig.colorbar(plot)
        plt.show()
