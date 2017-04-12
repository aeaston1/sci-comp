import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import pandas # for printing

# Number of discrete time and spatial steps
N = 100
T = 100
C, delta_x, delta_t = 1, 1/float(N), 0.01

# The transversal string position for all x-values at t = 0
def t0(n):
    x = n/float(N)
    return math.sin(2*math.pi*x)

# The first time derivative (transversal speed)
# for all x-values at t = 0
def t0_deriv(x):
    return 0

first, second = [], []
for n in range(N):
    first.append(t0(n))
    second.append(t0(n) + delta_t*t0_deriv(n))

M = [first, second]
D = C*(delta_t/float(delta_x))**2
for t in range(T-2):
    t = t + 2
    thisStep, lastStep, lastlastStep = [0], M[-1], M[-2]
    for n in range(N-2):
        n += 1
        thisStep.append(
            D*(lastStep[n-1]-2*lastStep[n]+lastStep[n+1]) \
            + 2*lastStep[n] - lastlastStep[n]
        )
    thisStep.append(0)
    M.append(thisStep)
    # print(pandas.DataFrame(M))

plt.matshow(M)
plt.show()
