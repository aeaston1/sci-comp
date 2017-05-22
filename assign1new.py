import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas # for printing matrices
import scipy.linalg

L = 1
delta_xy = L/50
width = round(L/delta_xy)
height = round(2*L/delta_xy)
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
        return coordinates, [0 for i in coordinates]

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
        else: expandVec.append(1)

    A = np.zeros((height+1, width+1))
    for n, coor in enumerate(coordinates):
        x, y = coor
        A[x,y] = vec[n]

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

    # The coordinate and index of the source are obtained
    source_x, source_y = round(2.0/delta_xy), round(2.0/delta_xy)
    print(source_x, source_y)
    source_n = (source_y)*(width + 1) + source_x
    sourceVec = np.array([1 if i == source_n else 0 for i in np.arange(totLen)])
    # The finite-difference matrix is obtained
    M = matrix(coordinates, boundary, totLen, source_n)

    # The matrix is altered to remove the empty rows and columns
    reduceM = reduceMatrix(M,boundary)

    plt.matshow(reduceM)
    plt.show()

    eigenValues, eigenVectors = scipy.linalg.eig(M)
    # idx = eigenValues.argsort()[::-1]
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    eigenVectors=eigenVectors.T
    print(eigenValues)
    for m,vec in enumerate(eigenVectors[:7]):
        thisVec = np.zeros((height+1, width+1))
        for n, coor in enumerate(coordinates):
            x, y = coor
            thisVec[y,x] = vec[n]
        vec = thisVec
        print(eigenValues[m])
        plt.matshow(vec)
        plt.show()
