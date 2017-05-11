import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import scipy as sp
import math
import pandas # for printing
import copy
import time
from numba import jit, autojit


#intiail conditions, denoted with "_i"
del_t = 1
del_x = 1
D_u_i = 0.16
D_v_i = 0.08
f_i = 0.035
k_i = 0.06
u_i = 0.5 #initially everywhere in the system
v_i = 0.25 #intialy in a small square in the centre of the system 0 elsewhere
p_i = 0 #initialise this to zero
n_rows_i = 100 #number of rows
n_cols_i = 100 #number of columns
Res = np.zeros((n_rows_i, n_cols_i), dtype=float)
P = np.zeros((n_rows_i, n_cols_i), dtype=float)
timesteps = 2
N = 100
coordinates = [(x,y+1) for x in range(N+1) for y in range(N-1)]
T = 100 #number of steps in interval 0->1
Cons_u = D_u_i*(del_t/float(del_x**2))
Cons_v = D_v_i*(del_t/float(del_x**2))

def fill_U(u):
    return np.full((n_rows_i+1, n_cols_i+1), u, dtype=float)

def fill_V(v):
    #create a square of chemical in the centre of the square
    V = np.zeros((n_rows_i+1, n_cols_i+1), dtype=float)
    for i in range(int(n_rows_i/2 - 2), int(n_rows_i/2 + 2)):
        for j in range(int(n_cols_i/2 - 2), int(n_cols_i/2 + 2)):
            V[i,j] = v
    return V

def fill_UV(u,v):
    for i in range(n_rows_i):
        for j in range(n_cols_i):
            if i >= (int(n_rows_i/2 -2)) and i <= (int(n_rows_i/2 + 2)):
                if j >= (int(n_cols_i/2 - 2)) and j <= (int(n_cols_i/2 + 2)):
                    Res[i][j] = np.array[(u,v)]
                else:
                    continue
            Res[i][j] = np.array[(u,0.0)]
    return Res

def u_to_v(u, v):
    new_v = u + 2*v
    return 3 * new_v

def v_to_p(v, p):
    p = copy.deepcopy(v)
    return p

def u_time(d_t, D_u, U, V, f, i,j):
    return d_t * (D_u * U[i,j]) - (U[i,j] * (V[i,j])**2) + (f * (1-U[i,j]))

def v_time(d_t, D_v, U, V, f, k, i, j):
    return d_t * (D_v * V[i,j]) + (U[i,j] * (V[i,j])**2) - (V[i,j] * (f + k))

def hessian(U):
    U_grad = np.gradient(U)
    hessian = np.empty((U.ndim, U.ndim) + U.shape, dtype=U.dtype)
    for k, grad_k in enumerate(U_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def utime(d_t, D_u, U, V, f):
    V_sq = np.matmul(V,V)
    du = d_t * ((D_u * hessian(U)) - (np.matmul(U, V_sq)) + (f * (1 - U)))
    return du

def vtime(d_t, D_v, U, V, f, k):
    V_sq = np.matmul(V,V)
    dv = d_t * ((D_v * hessian(V)) + (np.matmul(U, V_sq)) - ((f + k) * V))
    return dv

###############################################################################
# Build empty matrix
# M = np.zeros((N+1,N+1))
# for n in np.arange(N+1):
#     M[n,0] = 0
#     M[n,N] = 1
# M = np.matrix(M)
# newM = copy.deepcopy(M)
# def analytic(x, t):
#     c = 0
#     a = 2*math.sqrt(D*t)
#     for i in np.arange(1000):
#         c += (math.erfc((1-x+(2*i))/a) - math.erfc((1+x+(2*i))/a))
#     return c
# analytic_list = []
# for x in np.arange(0,N+1):
#     analytic_list.append(analytic(x/100.0, T/delta_t))

def grayscott_u(x,y,M_u,M_v):
    newU[x,y] = M_u[x,y] + Cons_u*( \
        M_u[x-1,y] + M_u[(x+1)%N,y] + M_u[x,y-1] + M_u[x,y+1] - 4*M_u[x,y]) \
        - M_u[x,y]*(M_v[x,y])**2 + f_i * (1 - M_u[x,y])

def grayscott_v(x,y,M_u,M_v):
    newV[x,y] = M_v[x,y] + Cons_v*( \
        M_v[x-1,y] + M_v[(x+1)%N,y] + M_v[x,y-1] + M_v[x,y+1] - 4*M_v[x,y]) \
        - M_u[x,y]*(M_v[x,y])**2 + f_i * (f_i + k_i) * M_v[x,y]

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


def simulation(totT, newu, u, Consu, newv, v, Consv):
    for t in np.arange(totT):
        newu = Consu*(TD*u+u*TD + Boundary*u) + (1-4*Consu)*u #- (u * v**2 + (f_i * (1 - u)))
        newv = Consv*(TD*v+v*TD + Boundary*v) + (1-4*Consv)*v #+ (u * v**2 - (f_i + k_i) * v)
        u = copy.deepcopy(newu)
        v = copy.deepcopy(newv)

        v[v.nonzero()] = ((2/3) * v[v.nonzero()]) + ((1/3) * u[v.nonzero()])

    return u,v


if __name__ == "__main__":
    # first
    # fill the U and V arrays with the initial values.
    # in one time step, react U with V and output to the resultant array
    # after the first time step the resultant array must be the only array to be worked on
    # potentially create an array of tuples, one element is the U value and one is the V value
    #TODO: check out the return dimensions of the hessian function
    # U = fill_U(u_i)
    # V = fill_V(v_i)
    # U = utime(del_t, D_u_i, U, V, f_i)
    # V = vtime(del_t, D_v_i, U, V, f_i, k_i)
    #
    # for t in range(timesteps):
    #     U = utime(del_t, D_u_i, U[0][0], V[0][0], f_i)
    #     V = vtime(del_t, D_v_i, U[0][0], V[0][0], f_i, k_i)
    #             V[i,j] = v_time(del_t, D_v_i, U, V, f_i, k_i, i, j)
    # fig, ax = plt.subplots(2)
    # cax0 = ax[0].imshow(U[0][0], interpolation='nearest', cmap=cm.afmhot)
    # cax1 = ax[1].imshow(V[0][0], interpolation='nearest', cmap=cm.afmhot)
    # plt.show()

    #second
    P = np.zeros((N,N))
    U = fill_U(u_i)
    V = fill_V(v_i)
    U = np.matrix(U)
    V = np.matrix(V)
    newU = copy.deepcopy(U)
    newV = copy.deepcopy(V)
    a = [1 for n in np.arange(N)]
    b = [0 for n in np.arange(N+1)]
    TD = tridiag(a,b,a)
    TD = np.matrix(TD)
    Boundary = np.zeros((N+1,N+1))
    # Boundary[0,N] = 1
    # Boundary[N,0] = 1
    # Boundary[0,0] = 1
    # Boundary[N,N] = 1
    Boundary = np.matrix(Boundary)
    print("Starting loop...")
    start = time.time()
    newU,newV = simulation(100, newU, U, Cons_u, newV, V, Cons_v)
    # simV = simulation(1, newV, V, Cons_v)
    end = time.time()
    print(end-start)

    newU = np.array(newU)
    newV = np.array(newV)
    fig, ax = plt.subplots(2)
    ax[0].matshow(newU)
    ax[1].matshow(newV)
    plt.xlabel("x value", fontsize=20)
    plt.ylabel("y value", fontsize=20)
    plt.show()
