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
Cons = D*(delta_t/float(delta_x)**2)

# Build empty matrix
M = np.zeros((N+1,N+1))
for n in range(N+1):
    M[n,0] = 0
    M[n,N] = 1
newM = copy.deepcopy(M)

def analytic(x, t):
    c = 0
    for i in range(1000):
        a = 2*math.sqrt(D*t)
        c += math.erfc((1-x+2*i)/a)
        c += math.erfc((1+x+2*i)/a)
    return c

for t in range(T):
    for x in range(N+1):
        for y in range(N-1):
            y += 1
            newM[x,y] = M[x,y] + Cons*(
                M[x-1,y] + M[(x+1)%N,y] + M[x,y-1] + M[x,y+1] - 4*M[x,y]
                )

    M = copy.deepcopy(newM)
print(pandas.DataFrame(M))

plt.matshow(M)
plt.show()
