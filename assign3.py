import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas # for printing matrices
import scipy.linalg
import scipy.sparse.linalg
import time

L = 4
delta_xy = L/20.0
width = int(L/delta_xy + 1)
height = int(L/delta_xy + 1)
isCircle = True

def getGrid(width, height, isCircle):
    '''
    The function collects all coordinates in the grid
    and checkes if they satisfy the boundary conditions
    '''
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
        i, j = coor
        if i*delta_xy == 0.6 and j*delta_xy == 1.2: print('FOUND IT')
        if boundary[n]:
            M.append([0 for i in np.arange(totLen)])
            continue
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

    M = np.matrix(M)
    return M/delta_xy**2

if __name__ == "__main__":
    totLen = (width+1)*(height+1)
    coordinates, boundary = getGrid(width, height, isCircle)
    M = matrix(coordinates, boundary, totLen)

    source_x, source_y = int(0.6/delta_xy), int(1.2/delta_xy)
    source_n = source_y*width + source_x
    print(source_n)

    vals, vecs = scipy.linalg.eig(M)
    for vec in vecs:
        thisVec = np.zeros((height+1, width+1))
        for n, coor in enumerate(coordinates):
            x, y = coor
            thisVec[x,y] = vec[n]
        vec = thisVec
    print(vals)
    plt.matshow(vec)
    plt.show()
