import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas # for printing matrices
import scipy.linalg

L = 4
delta_xy = L/20.0
width = round(2*L/delta_xy)
height = round(L/delta_xy)
isCircle = False

def getGrid(width, height, isCircle):
    '''
    The function collects all coordinates in the grid
    and checkes if they satisfy the boundary conditions
    '''
    coordinates = [
        (x,y) for y in np.arange(height+1) for x in np.arange(width+1)
        ]

    if not isCircle:
        boundary = []
        for x, y in coordinates:
            if x == 0 or x == width or y == 0 or y == height:
                boundary.append(1)
            else: boundary.append(0)
        return coordinates, boundary

    # Boundaries are more complicated for the circle!
    boundary = []
    half = (width)/2.0
    for x, y in coordinates:
        if (x-half)**2 + (y-half)**2 >= (half)**2:
            boundary.append(1)
        else: boundary.append(0)

    return coordinates, boundary

def matrix(coordinates, boundary, totLen, source_n):
    '''
    The matrix generation for the square matrix.
    Also soon to be for both matrices and the cirle.
    '''

    M = []
    for n, coor in enumerate(coordinates):
        i, j = coor
        # Any sources should only return themselves in the matrix
        if n == source_n:
            a = [0 for e in np.arange(totLen)]
            a[n] = 1
            M.append(a)
            continue
        # The boundary terms always return an array of zeros
        if boundary[n]:
            M.append([1 if e == n else 0 for e in np.arange(totLen)])
            continue
        # The finite difference scheme-matrix is build
        a = []
        for m, coor in enumerate(coordinates):
            k, l = coor
            if k == i and abs(l-j) == 1:
                a.append(1)
            elif abs(k-i) == 1 and l == j:
                a.append(1)
            else: a.append(0)
        for m, coor in enumerate(coordinates):
            k, l = coor
            if k == i and l == j:
                a[m] = -sum(a)
        M.append(a)

    return M

def toGrid(vec, boundary):
    '''
    The function takes a vector and transforms it to a grid
    '''
    A = np.zeros((width+1, height+1))
    for n, coor in enumerate(coordinates):
        x, y = coor
        A[x,y] = vec[n]

    return A

if __name__ == "__main__":
    totLen = (width+1)*(height+1)
    coordinates, boundary = getGrid(width, height, isCircle)

    # The coordinate and index of the source are obtained
    source_x, source_y = round(0.6/delta_xy), round(1.2/delta_xy)
    source_n = (source_y)*(width + 1) + source_x
    sourceVec = np.array([1 if i == source_n else 0 for i in np.arange(totLen)])

    # The finite-difference matrix is obtained
    M = matrix(coordinates, boundary, totLen, source_n)

    # The equation is solved
    result = scipy.linalg.solve(M, sourceVec)
    result = toGrid(result, boundary)

    result = np.matrix(result)
    result[result==0] = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot = ax.imshow(result, extent=[0, 4, 0, 4], origin='lower')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    fig.colorbar(plot)
    plt.show()
