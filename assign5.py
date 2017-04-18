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
newM = copy.deepcopy(M)

i = 0
while (i < maxiters):
    i += 1
    coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
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
        print(i) # Shows break-off iteration
        break
    M = copy.deepcopy(newM)
# print(pandas.DataFrame(M))

plt.matshow(M)
plt.show()
