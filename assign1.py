import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas # for printing matrices
import copy
import scipy.linalg
import scipy.sparse.linalg
import time

N = 10
# Set up parameters
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
delta_xy = 1/float(N)
T = 1
growthSteps = 6001
epsilon = 0.005 # Break-off value for convergence
omega = 1.5 # For SOR algoritm
u_i = 0.5 # fill value for the membrane shapes
c = 1
L = 1
del_t = 1
del_x = float(L/N)
del_y = float(1 * L/N) # for square, factor is 1. for rectangle, factor is 2.
Cons_u = c**2 *(del_t/float(del_x**2))
diag_num  = (-2./float(del_x)**2) + (-2./float(del_y)**2)

#TODO: notes
# double check I have the rectangle implemented correctly --> done, check with kk
# implement the circular properly
# review eigen vectors
# review each of the eigenvalue methods in P
# review what the size L is, pretty sure it is the matrix size -->matrix size
# what are the discretisation steps? --> del_x and del_y
# time dependent solutions, eigenfrequencies (-values) behave in time

def matrix():
    '''
    The matrix generation for the square matrix.
    Also soon to be for both matrices and the cirle.
    '''
    a = np.diag([diag_num for n in np.arange((N-1)**2)])
    bi = np.diag([
        1/del_y**2 if (n+1)%(N-1) != 0 else 0
        for n in np.arange((N-1)**2-1)]
        , 1)
    bj = np.diag([
        1/del_y**2 if (n+1)%(N-1) != 0 else 0
        for n in np.arange((N-1)**2-1)]
        , -1)
    ci = np.diag([1/del_x**2 for n in np.arange((N-1)**2-N-1)], -N-1)
    cj = np.diag([1/del_x**2 for n in np.arange((N-1)**2-N-1)], N+1)
    M = np.matrix(ci + cj + bi + bj + a)
    return M

def matrix_rectangle():
    '''
    Matrix generation for the rectangle, which is currently wrong.
    '''
    # the range of points for the rectangle is doubled
    # then the array is split row-wise to generate the rectangular matrix
    a = np.diag([diag_num for n in np.arange(2*(N-1)**2)])
    b = np.diag([
        1 if (n+1)%(N-1) != 0 else 0
        for n in np.arange(2*(N-1)**2-1)]
        , 1)
    c = np.diag([1 for n in np.arange(2*(N-1)**2-N-1)], N+1)
    M = np.matrix(c.T+b.T+a+b+c)
    #the return index is the split square array to rectangular shape
    return M[:((N-1)**2)]

def simulation(totT, mat_u, M, Consu, shape):
    '''
    totT : number of time iterations
    mat_u : the matrix membrane
    M : the matrix of the eigenvalue problem
    Consu : the constant values out the front of the wave equation
    Square : indicating whether the matrix will be square or not
    '''
    if shape=="square":
        for t in np.arange(totT):
            newu = Consu*(M*mat_u+mat_u*M + Boundary_square*mat_u)
            mat_u = copy.deepcopy(newu)
    if shape=="rectangle":
        for t in np.arange(totT):
            newu = Consu*(M.T*mat_u+mat_u.T*M + Boundary_rectangle.T*mat_u)
            mat_u = copy.deepcopy(newu[:((N-1)**2)]) #ensure the new shape is copied
    return newu

def fill_square(u):
    '''
    fill a square matrix with some values.
    (not sure of the point of this)
    '''
    return np.full(((N-1)**2, (N-1)**2), u, dtype=float)

def fill_rectangle(u):
    '''
    fill a rectangle with some values.
    (not sure of the point of this)
    '''
    #split by row, hence the indexing at the end
    return np.full((2*(N-1)**2, 2*(N-1)**2), u, dtype=float)[:((N-1)**2)]

def boundary_square():
    '''
    boundary conditions for the square.
    '''
    Boundary = np.zeros(((N-1)**2,(N-1)**2))
    Boundary[0,N] = 1
    Boundary[N,0] = 1
    Boundary[0,0] = 1
    Boundary[N,N] = 1
    return np.matrix(Boundary)

def boundary_rectangle():
    '''
    boundary conditions for a rectangle
    '''
    Boundary = np.zeros(((N-1)**2,2*(N-1)**2))
    Boundary[0,N] = 1
    Boundary[N,0] = 1
    Boundary[0,0] = 1
    Boundary[N,N] = 1
    return np.matrix(Boundary)

if __name__ == "__main__":
    # boundaries
    # Boundary_square = boundary_square()
    # Boundary_rectangle = boundary_rectangle()
    print('Starting matrix creation of size : %d' % N)
    start = time.time()
    M_square = matrix()
    end = time.time()
    print('Matrix creation took : %fs' % (end-start))
    print(' ')
    print('Starting scipy.linalg.eigh() creation...')
    start = time.time()
    eig2_vals,eig2_vecs = scipy.linalg.eigh(M_square)
    end = time.time()
    print('scipy.linalg.eigh() creation took : %fs' % (end-start))
    print(' ')

    print(eig2_vals)
    fig, ax = plt.subplots()
    ax.plot(eig2_vecs[0])
    ax.plot(eig2_vecs[1])
    ax.plot(eig2_vecs[2])
    ax.plot(eig2_vecs[3])
    ax.plot(eig2_vecs[4])
    ax.plot(eig2_vecs[5])
    ax.plot(eig2_vecs[6])
    ax.plot(eig2_vecs[7])
    ax.plot(eig2_vecs[8])
    ax.plot(eig2_vecs[9])
    plt.show()
    #other eigenvalues claculation methods
    # print('Starting scipy.linalg.eig() creation...')
    # start = time.time()
    # eig1_vals,eig1_vecs = scipy.linalg.eig(M_square, right=True)
    # end = time.time()
    # print('scipy.linalg.eig() creation took : %fs' % (end-start))
    # print(' ')
    # print('Starting numpy.linalg.eig() creation...')
    # start = time.time()
    # eig3_vals,eig3_vecs = np.linalg.eig(M_square)
    # end = time.time()
    # print('numpy.linalg.eig() creation took : %fs' % (end-start))
    # print(' ')
    # print('Starting scipy.sparse.linalg.eigs() creation...')
    # start = time.time()
    # eig4_vals,eig4_vecs = scipy.sparse.linalg.eigs(M_square)
    # end = time.time()
    # print('scipy.sparse.linalg.eigs() creation took : %fs' % (end-start))
    # print(' ')

    # generating some membranes : think this is pointless
    # U_square = fill_square(u_i)
    # U_rectangle = fill_rectangle(u_i)
    # newU_square = copy.deepcopy(U_square)
    # newU_rectangle = copy.deepcopy(U_rectangle)
    # mat_U_square = np.matrix(U_square)
    # mat_U_rectangle = np.matrix(U_rectangle)
    # newU_square = simulation(T, mat_U_square, M_square, Cons_u)
    # newU_rectangle = simulation(T, mat_U_rectangle, M_rectangle, Cons_u, Square=False)
    # fig, ax = plt.subplots()
    # ax.matshow(M_square)
    # ax[0].matshow(newU_square)
    # ax[1].matshow(newU_rectangle)
    # plt.show()
