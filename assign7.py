import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import math
import pandas # for printing
import copy

# Set up parameters
N = 50
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
Cons, delta_xy = 1/4.0, 1/float(N)
maxiters = 1000
epsilon = 0.001 # Break-off value for convergence
omega = 1.7 # For SOR algoritm

def newMatrix():
    # Build empty matrix
    M = np.zeros((N+1,N+1))
    for n in range(N+1):
        M[n,0] = 0
        M[n,N] = 1
    return M

def analytic(y):
    return y

def Jacobi(M):
    k = 0
    while (k < maxiters):
        k += 1

        for x, y in coordinates:
            newM[x,y] = Cons*(
                M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1]
                )

        # Determine if system is converged
        difference = abs(M - newM)
        for x, y in coordinates:
            delta = difference[x,y]
            if delta > epsilon:
                break
        else:
            # obscure python code -> for-else statement,
            # fires else when break is not fired
            print("Jacobi terminated in iteration ",k)
            break

        M = copy.deepcopy(newM)

    return M, k

def GaussSeidel(M):
    k  = 0
    while (k < maxiters):
        c = 0
        k += 1

        for x, y in coordinates:
            newValue = Cons*(M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1])
            delta = abs(newValue - M[x,y])
            M[x,y] = newValue

        # Determine if system is converged
            if delta > epsilon:
                c += 1
        if c == 0:
            print("Gauss-Seidel terminated in iteration ",k) #add timer?
            break

    return M, k

def SOR(M, omega):
    k  = 0
    while (k < maxiters):
        c = 0
        k += 1

        for x, y in coordinates:
            newValue = omega/4.0*(M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1]) \
                            + (1 - omega)*M[x,y]
            delta = abs(newValue - M[x,y])
            M[x,y] = newValue

        # Determine if system is converged
            if delta > epsilon:
                c += 1
        if c == 0:
            print("Gauss-Seidel terminated in iteration ",k) #add timer?
            break

    return M, k

if __name__ == "__main__":
    # Question G
    JacobiSolution = Jacobi(newMatrix())[0]
    GaussSeidelSolution = GaussSeidel(newMatrix())[0]
    SORSolution = SOR(newMatrix(), omega)[0]
    analyticSolution = [analytic(y+1) for y in range(N-1)]

    plt.plot(JacobiSolution[0])
    plt.plot(GaussSeidelSolution[0])
    plt.plot(SORSolution[0])
    plt.plot(analyticSolution)
    plt.show()

    # Question H
    deltas = [d**-(i+1) for i in range(4)]
    iterations = []
    for delta in deltas:
        iterations += Jacobi(newMatrix())[1]
        iterations += GaussSeidel(newMatrix())[1]
        for omega in [1.7, 1.8, 1.9, 2.0]:
            iterations += SOR(newMatrix(), omega)[1]

    # Plot number of iterations as function of delta

    # Question I
    def minROS(omega):
        M = newMatrix()
        return ROS(M, omega)

    res = minimize(minROS, omega, bounds = (1, 1.99))
    optimalOmega = res.x

# plt.matshow(M)
# plt.show()
