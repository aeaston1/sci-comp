import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import pandas # for printing

# Number of discrete time and spatial steps
N= 100
T = 100
C, delta_x, delta_t = 1, 1/float(N), 0.01

# The transversal string position for all x-values at t = 0
# Question 1
def t0(n):
    x = n/float(N)
    return math.sin(2*math.pi*x)

# # The transversal string position for all x-values at t = 0
# # Question 2
# def t0(n):
#     x = n/float(N)
#     return math.sin(5*math.pi*x)

# # The transversal string position for all x-values at t = 0
# # Question 3
# def t0(n):
#     x = n/float(N)
#     if x <= 1/5 or x >= 2/5:
#         return 0
#     return math.sin(5*math.pi*x)

# The first time derivative (transversal speed)
# for all x-values at t = 0
def t0_deriv(x):
    return 0

M = np.zeros((T,N))
D = C*(delta_t/float(delta_x))**2
# Defining initial conditions in matrix
for n in range(N):
    M[0,n] = t0(n)
    M[1,n] = t0(n) + delta_t*t0_deriv(n)

for t in range(T-2):
    t = t + 2
    M[t,0], M[t,N-1] = 0, 0
    for n in range(N-2):
        n += 1
        M[t,n] = (
            D*(M[t-1,n-1]-2*M[t-1,n]+M[t-1,n+1]) \
            + 2*M[t-1,n] - M[t-2,n]
        )
print(pandas.DataFrame(M))

plt.matshow(M)
plt.show()
