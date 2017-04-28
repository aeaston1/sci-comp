import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import pandas # for printing
import copy
import numpy as np
import time
from numba import jit, autojit

# Set up parameters
N = 100
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
T = 100 #number of steps in interval 0->1
D, delta_x, delta_t = 0.001, 1/float(N), 1/float(T)
Cons = D*(delta_t/float(delta_x**2))

# Build empty matrix
M = np.zeros((N+1,N+1))
for n in np.arange(N+1):
    M[n,0] = 0
    M[n,N] = 1
M = np.matrix(M)
newM = copy.deepcopy(M)

def analytic(x, t):
    c = 0
    a = 2*math.sqrt(D*t)
    for i in np.arange(1000):
        c += (math.erfc((1-x+(2*i))/a) - math.erfc((1+x+(2*i))/a))
    return c

analytic_list = []
for x in np.arange(0,N+1):
    analytic_list.append(analytic(x/100.0, T/delta_t))

def iteration(x,y,M):
    #print(type(x), type(y), type(M))
    newM[x,y] = M[x,y] + Cons*(
        M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1] - 4*M[x,y])

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

a = [1 for n in np.arange(N)]
b = [0 for n in np.arange(N+1)]
TD = tridiag(a,b,a)
TD = np.matrix(TD)
Boundary = np.zeros((N+1,N+1))
Boundary[N,0] = 1
Boundary[0,N] = 1
Boundary = np.matrix(Boundary)

def simulation(totT, newM, M):
    for t in np.arange(totT):#
        newM = Cons*(TD*M+M*TD + Boundary*M) + (1-4*Cons)*M

        newM[:,-1] = 1
        newM[:,0] = 0
        if t%10000 == 0:
            print(t)

        if t == 0:
            simulated0 = copy.deepcopy(newM)
        if t == 100:
            simulated100 = copy.deepcopy(newM)
        if t == 1000:
            simulated1000 = copy.deepcopy(newM)
        if t == 10000:
            simulated10000 = copy.deepcopy(newM)
        M = copy.deepcopy(newM)

    return M, simulated0, simulated100, simulated1000, simulated10000
if __name__ == "__main__":
    # plot for part E
    print("Starting loop...")
    start = time.time()
    M, simulated0, simulated100, simulated1000, \
    simulated10000 = simulation(100000, newM, M)
    end = time.time()
    print(end-start)
    M = np.array(M)
    simulated0 = np.array(simulated0)
    simulated100 = np.array(simulated100)
    simulated1000 = np.array(simulated1000)
    simulated10000 = np.array(simulated10000)

    x = np.linspace(0,101,N+1)
    fig, ax = plt.subplots()
    analytic, = ax.plot(x, analytic_list, label='Analytic Solution', linewidth=2)
    sim0, = ax.plot(x, simulated0[0], label='Simulation t=0', linewidth=2)
    sim100, = ax.plot(x, simulated100[0], label='Simulation t=1', linewidth=2)
    sim1000, = ax.plot(x, simulated1000[0], label='Simulation t=10', linewidth=2)
    sim10000, = ax.plot(x, simulated10000[0], label='Simulation t=100', linewidth=2)
    sim1000000, = ax.plot(x, M[0], label='Simulation t=1000', linewidth=2)
    ax.legend(loc='upper left', fontsize=20)
    plt.xlabel("x value", fontsize=20)
    plt.ylabel("Concentration", fontsize=20)

    plt.plot()


    # array rotations
    M = np.rot90(M)
    simulated10000 = np.rot90(simulated10000)
    simulated1000 = np.rot90(simulated1000)
    simulated100 = np.rot90(simulated100)
    # plots for part F
    fig, axarr = plt.subplots(ncols=2, nrows=2)
    axarr[0,0].matshow(M)
    axarr[0,0].set_title("t = 1000",fontsize=30)
    axarr[0,1].matshow(simulated10000)
    axarr[0,1].set_title("t = 100",fontsize=30)
    axarr[1,0].matshow(simulated1000)
    axarr[1,0].set_title("t = 10",fontsize=30)
    axarr[1,0].xaxis.set_ticks_position('bottom')
    axarr[1,1].matshow(simulated100)
    axarr[1,1].set_title("t = 1",fontsize=30)
    axarr[1,1].xaxis.set_ticks_position('bottom')
    axarr[1,0].set_xlabel("x value", fontsize=20)
    axarr[1,1].set_xlabel("x value", fontsize=20)
    axarr[1,0].set_ylabel("y value", fontsize=20)
    axarr[0,0].set_ylabel("y value", fontsize=20)

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], fontsize=20)
    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], fontsize=20)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.show()
