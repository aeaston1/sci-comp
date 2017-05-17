import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas # for printing matrices
import copy
import scipy.linalg
import scipy.sparse.linalg

N = 4
# Set up parameters
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
delta_xy = 1/float(N)
T = 100
growthSteps = 6001
epsilon = 0.005 # Break-off value for convergence
omega = 1.5 # For SOR algoritm
u_i = 0.5 # fill value for the membrane shapes
c = 1
del_t = 1
del_x = 1
Cons_u = c**2 *(del_t/float(del_x**2))
#TODO: don't use -4, use the equation for -4 below.
#interval/ (length/breadth)
def matrix_square():
    a = np.diag([-4 for n in np.arange((N-1)**2)])
    b = np.diag([
        1 if (n+1)%(N-1) != 0 else 0
        for n in np.arange((N-1)**2-1)]
        , 1)
    c = np.diag([1 for n in np.arange((N-1)**2-N-1)], N+1)
    M = np.matrix(c.T+b.T+a+b+c)
    return M

def matrix_rectangle():
    # the range of points for the rectangle is doubled
    # then the array is split row-wise to generate the rectangular matrix
    a = np.diag([-4 for n in np.arange(2*(N-1)**2)])
    b = np.diag([
        1 if (n+1)%(N-1) != 0 else 0
        for n in np.arange(2*(N-1)**2-1)]
        , 1)
    c = np.diag([1 for n in np.arange(2*(N-1)**2-N-1)], N+1)
    M = np.matrix(c.T+b.T+a+b+c)
    #the return index is the split square array to rectangular shape
    return M[:((N-1)**2)]

def simulation(totT, mat_u, M, Consu, Square=True):
    '''
    totT : number of time iterations
    mat_u : the matrix membrane
    M : the matrix of the eigenvalue problem
    Consu : the constant values out the front of the wave equation
    Square : indicating whether the matrix will be square or not
    '''
    if Square:
        for t in np.arange(totT):
            newu = Consu*(M*mat_u+mat_u*M + Boundary_square*mat_u)
            mat_u = copy.deepcopy(newu)
    else:
        for t in np.arange(totT):
            newu = Consu*(M.T*mat_u+mat_u.T*M + Boundary_rectangle.T*mat_u)
            mat_u = copy.deepcopy(newu[:((N-1)**2)]) #ensure the new shape is copied
    return newu

def fill_square(u):
    return np.full(((N-1)**2, (N-1)**2), u, dtype=float)

def fill_rectangle(u):
    #split by row, hence the indexing at the end
    return np.full((2*(N-1)**2, 2*(N-1)**2), u, dtype=float)[:((N-1)**2)]

def boundary_square():
    Boundary = np.zeros(((N-1)**2,(N-1)**2))
    Boundary[0,N] = 1
    Boundary[N,0] = 1
    Boundary[0,0] = 1
    Boundary[N,N] = 1
    return np.matrix(Boundary)

def boundary_rectangle():
    Boundary = np.zeros(((N-1)**2,2*(N-1)**2))
    Boundary[0,N] = 1
    Boundary[N,0] = 1
    Boundary[0,0] = 1
    Boundary[N,N] = 1
    return np.matrix(Boundary)

if __name__ == "__main__":
    # boundaries
    Boundary_square = boundary_square()
    Boundary_rectangle = boundary_rectangle()
    M_square = matrix_square()
    M_rectangle = matrix_rectangle()
    # w,vl,vr = scipy.linalg.eig(M_square, left=True, right=True)
    # eigh_vals, eigh_vec = scipy.linalg.eigh(M_square)
    # eigs_vals, eigs_vec = scipy.sparse.linalg.eigs(M)

    U_square = fill_square(u_i)
    U_rectangle = fill_rectangle(u_i)
    newU_square = copy.deepcopy(U_square)
    newU_rectangle = copy.deepcopy(U_rectangle)
    mat_U_square = np.matrix(U_square)
    mat_U_rectangle = np.matrix(U_rectangle)
    # temp = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
    # print(temp)
    # print(temp[:2]) # split by row
    # print(temp[:,0:2]) # split by column

    print(M_rectangle)
    stop
    newU_square = simulation(T, mat_U_square, M_square, Cons_u)
    newU_rectangle = simulation(T, mat_U_rectangle, M_rectangle, Cons_u, Square=False)

    fig, ax = plt.subplots(2)
    ax[0].matshow(newU_square)
    ax[1].matshow(newU_rectangle)
    plt.show()
