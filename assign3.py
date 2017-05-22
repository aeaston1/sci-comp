import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas # for printing matrices
import copy
import scipy.linalg
import scipy.sparse.linalg
import time

width = 7
height = 3
isCircle = False

# delta_xy = 1/float(N)
# T = 1
# u_i = 0.5 # fill value for the membrane shapes
# c = 1
# L = 1
# del_t = 1

def getGrid(width, height, isCircle):
    coordinates = [
        (x,y) for x in np.arange(width+1) for y in np.arange(height+1)
        ]

    if not isCircle:
        return coordinates, [0 for i in coordinates]

    # Boundaries are more complicated for the circle!
    boundary = []
    half = (width)/2.0
    for x, y in coordinates:
        if (x-half)**2 + (y-half)**2 >= (half)**2:
            boundary.append(1)
        else: boundary.append(0)

    return coordinates, boundary

def matrix(coordinates, boundary, totLen):
    '''
    The matrix generation for the square matrix.
    Also soon to be for both matrices and the cirle.
    '''

    M = []
    for n, coor in enumerate(coordinates):
        if boundary[n]:
            M.append([0 for i in np.arange(totLen)])
            continue
        i, j = coor
        a = []
        for m, coor in enumerate(coordinates):
            k, l = coor
            if k == i             and l == j:
                a.append(-4)
            elif k == i           and abs(l-j) == 1   and not boundary[m]:
                a.append(1)
            elif abs(k-i) == 1    and l == j          and not boundary[m]:
                a.append(1)
            else: a.append(0)

        M.append(a)

    return np.matrix(M)

# def simulation(totT, mat_u, M, Consu, shape):
#     '''
#     totT : number of time iterations
#     mat_u : the matrix membrane
#     M : the matrix of the eigenvalue problem
#     Consu : the constant values out the front of the wave equation
#     Square : indicating whether the matrix will be square or not
#     '''
#     if shape=="square":
#         for t in np.arange(totT):
#             newu = Consu*(M*mat_u+mat_u*M + Boundary_square*mat_u)
#             mat_u = copy.deepcopy(newu)
#     if shape=="rectangle":
#         for t in np.arange(totT):
#             newu = Consu*(M.T*mat_u+mat_u.T*M + Boundary_rectangle.T*mat_u)
#             mat_u = copy.deepcopy(newu[:((N-1)**2)]) #ensure the new shape is copied
#     return newu

if __name__ == "__main__":
    totLen = (width+1)*(height+1)
    coordinates, boundary = getGrid(width, height, isCircle)
    print('  '+str(boundary))
    M = matrix(coordinates, boundary, totLen)
    plt.matshow(M)
    plt.show()
