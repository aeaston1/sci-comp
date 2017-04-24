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

    M = copy.deepcopy(newM)

analytic_list = []
for x in range(0,N+1):
    x = x/N
    analytic_list.append(analytic(x, T/delta_t))

x= np.linspace(0,101,N+1)
fig, ax = plt.subplots()
analytic, = ax.plot(x, analytic_list, label='Analytic Solution', linewidth=2)
simulated0, = ax.plot(x, simulated0[0], label='Simulation t=0', linewidth=2)
simulated1, = ax.plot(x, simulated1[0], label='Simulation t=0.001', linewidth=2)
simulated10, = ax.plot(x, simulated10[0], label='Simulation t=0.01', linewidth=2)
simulated100, = ax.plot(x, simulated100[0], label='Simulation t=0.1', linewidth=2)
simulated1000, = ax.plot(x, M[0], label='Simulation t=1', linewidth=2)
ax.legend(loc='upper left', fontsize=30)
plt.plot()

#plt.matshow(M)
plt.show()
