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

# Build empty matrix
M = np.zeros((N+1,N+1))
for n in range(N+1):
    M[n,0] = 0
    M[n,N] = 1
newM = copy.deepcopy(M)

for i in range(1):
# Do we need multiple iterations?
# We do if we want to find the steady state right?
    for x in range(N+1):
        for y in range(N-1):
            y += 1
            newM[x,y] = Cons*(
                M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1]
                )

    M = copy.deepcopy(newM)
print(pandas.DataFrame(M))

plt.matshow(M)
plt.show()
