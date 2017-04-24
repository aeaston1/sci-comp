import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import pandas # for printing
import copy
import numpy as np

# Set up parameters
N = 100
T = 1000 #number of steps in interval 0->1
D, delta_x, delta_t = 0.01, 1/float(N), 0.001
Cons = D*(delta_t/float(delta_x)**2)

# Build empty matrix
M = np.zeros((N+1,N+1))
for n in range(N+1):
    M[n,0] = 0
    M[n,N] = 1
newM = copy.deepcopy(M)

def analytic(x, t):
    c = 0
    a = 2*math.sqrt(D*t)
    for i in range(1000):
        c += (math.erfc((1-x+(2*i))/a) - math.erfc((1+x+(2*i))/a))
    return c

analytic_list = []
for x in range(0,N+1):
    analytic_list.append(analytic(x/100.0, T/delta_t))

for t in range(1000):
    for x in range(N+1):
        for y in range(N-1):
            y += 1
            newM[x,y] = M[x,y] + Cons*(
                M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1] - 4*M[x,y]
                )
    if t == 0:
        simulated0 = copy.deepcopy(newM)
    if t == 1:
        simulated1 = copy.deepcopy(newM)
    if t == 10:
        simulated10 = copy.deepcopy(newM)
    if t == 100:
        simulated100 = copy.deepcopy(newM)
    if t == 500:
        simulated500 = copy.deepcopy(newM)

    M = copy.deepcopy(newM)


# plot for part E
'''
x = np.linspace(0,101,N+1)
fig, ax = plt.subplots()
analytic, = ax.plot(x, analytic_list, label='Analytic Solution', linewidth=2)
simulated0, = ax.plot(x, simulated0[0], label='Simulation t=0', linewidth=2)
simulated1, = ax.plot(x, simulated1[0], label='Simulation t=0.001', linewidth=2)
simulated10, = ax.plot(x, simulated10[0], label='Simulation t=0.01', linewidth=2)
simulated100, = ax.plot(x, simulated100[0], label='Simulation t=0.1', linewidth=2)
simulated1000, = ax.plot(x, M[0], label='Simulation t=1', linewidth=2)
ax.legend(loc='upper left')
plt.plot()
'''
# array rotations
M = np.rot90(M)
simulated500 = np.rot90(simulated500)
simulated100 = np.rot90(simulated100)
simulated10 = np.rot90(simulated10)
simulated1 = np.rot90(simulated1)
simulated0 = np.rot90(simulated0)
# plots for part F
fig, axarr = plt.subplots(ncols=2, nrows=3)
axarr[0,0].matshow(M)
axarr[0,0].set_title("t = 1",fontsize=30)
axarr[0,1].matshow(simulated500)
axarr[0,1].set_title("t = 0.5",fontsize=30)
axarr[1,0].matshow(simulated100)
axarr[1,0].set_title("t = 0.1",fontsize=30)
axarr[1,1].matshow(simulated10)
axarr[1,1].set_title("t = 0.01",fontsize=30)
axarr[2,0].matshow(simulated1)
axarr[2,0].set_title("t = 0.001",fontsize=30)
axarr[2,0].xaxis.set_ticks_position('bottom')
axarr[2,1].matshow(simulated0)
axarr[2,1].set_title("t = 0",fontsize=30)
axarr[2,1].xaxis.set_ticks_position('bottom')
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
plt.setp([a.get_xticklabels() for a in axarr[2, :]], fontsize=20)
plt.setp([a.get_yticklabels() for a in axarr[:, 0]], fontsize=20)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()
