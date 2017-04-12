#!/usr/bin/python

# Plot the analytic solution of the heat equation,
# Initial conditions: c(x,t) = 0 for 0 <= x < 1    at t = 0
# Boundary condtions: c(0,t) = 0
#                     c(1,t) = 1   
# see lecture slides, 'Exact solutions 3'

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

D = 1

# analytc expression for the concentration c(x, t)
# from the lecture slides
def c(x,t):
    N = 30               # upper limit for the sum
    c = 0
    d = 2*np.sqrt(D*t)
    for i in range (0,N):
        c += erfc((1-x+2*i) / d) - erfc((1+x+2*i) / d)
        #                        ^ note sign here
    return c

# apply c on a vector
def C(X, t):
    C = [c(x,t) for x in X]
    return C

# add a plot at time t
def draw(t):
    X = np.linspace (-1.5, 1.5, 100)
    Y = C(X, t)
    plt.plot(X,Y,'-',label=str(t))

for t in [.001, .01, .1, 1, 10]:
    draw(t)


plt.legend(loc='upper left')
plt.show()


# How this analytic solution can be obtained
#
# 1) erfc(x/2sqrt(Dt)) solves the heat equation
# this can be seen with a Laplace transform (often shown for erf, not erfc).
# The sum is constructed to saisfy the boundary conditions.
# The heat equation is linear -> a sum of solutions is a solution (superposition principle)
#
# 2) use the "method of reflection" to add up erfc() solutions
# The constructed expression is an odd function (around 0),
# and also odd around the point (1,1) -> the solution will keep this property
# for all times -> the solution will pass through (0,0) and (1,1) for all times
# -> the boundary conditions are satisfied
#
# http://math.tut.fi/~piche/pde/notes05.pdf
# http://www.ewp.rpi.edu/hartford/~wallj2/CHT/Notes/ch05.pdf

