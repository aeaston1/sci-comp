import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import pandas # for printing
import copy

# Set up parameters
N = 100
T = 200
D, delta_x, delta_t = 0.01, 1/float(N), 0.001
Cons = 1/4
maxiters = 1000
epsilon = 0.001 #break-off value for Jacobi iteration

# Build empty matrix
M = np.zeros((N+1,N+1))
for n in range(N+1):
    M[n,0] = 0
    M[n,N] = 1

i  = 0
while (i < maxiters):
    c = 0
    i += 1
    coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
    for x, y in coordinates:
        newValue = Cons*(M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1])
        delta = abs(newValue - M[x,y])
        M[x,y] = newValue
    # Determine if system is converged
        if delta > epsilon:
            c += 1
    if c == 0:
        print("Terminated in iteration ",i) #add timer?
        break
# print(pandas.DataFrame(M))

plt.matshow(M)
plt.show()
