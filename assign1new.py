import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas # for printing matrices
import scipy.linalg

L = 1
delta_xy = L/50.
width = round(L/delta_xy)
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

def matrix(coordinates, boundary, totLen):
    '''
    The matrix generation for the square matrix and circle
    '''

    M = []
    for n, coor in enumerate(coordinates):
        i, j = coor
        # The boundary terms always return an array of zeros
        if boundary[n]:
            M.append([0 for e in np.arange(totLen)])
            continue
        # The finite difference scheme-matrix is build
        a = []
        for m, coor in enumerate(coordinates):
            k, l = coor
            if k == i and abs(l-j) == 1 and not boundary[m]:
                a.append(1)
            elif abs(k-i) == 1 and l == j and not boundary[m]:
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
    expandVec = []
    i = 0
    for isBoundary in boundary:
        if not isBoundary:
            expandVec.append(vec[i])
            i += 1
        else: expandVec.append(0)

    A = np.zeros((height+1, width+1))
    for n, coor in enumerate(coordinates):
        x, y = coor
        A[x,y] = expandVec[n]

    return A

def reduceMatrix(M, boundary):
    '''
    Empty rows and column are removed, so a non-singular matrix is obtained
    '''
    reduceM = []
    for i, row in enumerate(M):
        if boundary[i]: continue
        a = []
        for j, value in enumerate(row):
            if boundary[j]: continue
            a.append(value)
        reduceM.append(a)

    return np.matrix(reduceM)

if __name__ == "__main__":
    totLen = (width+1)*(height+1)
    coordinates, boundary = getGrid(width, height, isCircle)

    # The finite-difference matrix is obtained
    M = matrix(coordinates, boundary, totLen)
    reduceM = reduceMatrix(M, boundary)

    # The matrix is altered to remove the empty rows and columns
    reduceM = reduceMatrix(M,boundary)

    # plt.matshow(reduceM)
    # plt.show()

    eigenValues, eigenVectors = scipy.linalg.eig(M)
    eigenVectors=eigenVectors.T
    print(eigenValues)
    plt.plot(eigenValues)
    plt.show()
    # for m,vec in enumerate(eigenVectors[:7]):
    #     thisVec = np.zeros((height+1, width+1))
    #     for n, coor in enumerate(coordinates):
    #         x, y = coor
    #         thisVec[y,x] = vec[n]
    #     vec = thisVec
    #     print(eigenValues[m])
    #     plt.matshow(vec)
    #     plt.show()
