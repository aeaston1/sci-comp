import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas # for printing
import copy
from  matplotlib.animation import FuncAnimation

# Set up parameters
N = 10
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
delta_xy = 1/float(N)
maxiters = 150
growthSteps = 15
eden = 1.0
epsilon = 0.005 # Break-off value for convergence
omega = 1.0 # For SOR algoritm

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
    # O[int(N/2.0)+1,1] = 1
    # O[int(N/2.0)+2,1] = 1
    # O[int(N/2.0)+2,2] = 1
    return np.matrix(O)

def SOR(M, O, epsilon, omega):
    k  = 0
    while (k < maxiters):
        k += 1

        # Execute iteration step and reset sinks and sources
        newM = (omega/4)*(A*M + M*A + B*M) + (1-omega)*M
        newM = setSinksSources(newM, O)

        # Check is algorithm is converged
        D = abs(newM - M)
        D[abs(D) < epsilon] = 0
        if np.sum(D) == 0:
            print("SOR terminated in iteration ",k) #add timer?
            break

        # Prepare for next iteration
        M = copy.deepcopy(newM)

    return newM, k

def doGrowthStep(M, O):
    # Get growth sites
    S = A*O + O*A
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
            # S[x,y] = np.sum([M[i,j]
            #             for i in np.arange(x-1,(x+2)%N)
            #             for j in np.arange(y-1,y+2)
            #             ])

    return M, O



# def init():
#     plot.set_data(data[0])
#     return plot
#
# def update(j):
#     plot.set_data(data[j])
#     return [plot]

if __name__ == "__main__":
    # Question K
    O = newSeed()
    M = newMatrix(O)
    for g in np.arange(growthSteps):
        plt.matshow(M)
        plt.show()
        M, k = SOR(M, O, epsilon, omega)
        M, O = doGrowthStep(M, O)
    plt.matshow(M)
    plt.show()
    #
    # n_frames = 3 #Numero de ficheros que hemos generado
    # data = np.empty(n_frames, dtype=object) #Almacena los datos
    #
    # #Leer todos los datos
    # for k in range(n_frames):
    #     data[k] = np.loadtxt("frame"+str(k))
    #
    #
    # fig = plt.figure()
    # plot =plt.imshow(data[0])
    #
    #
    # anim = FuncAnimation(fig, update, init_func = init, frames=n_frames, interval = 30, blit=True)
    #
    # plt.show()
