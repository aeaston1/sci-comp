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
import random
import warnings

del_t = 1
del_x = 1
d={}
# parameter sets
D_u_i1, D_v_i1, f_i1, k_i1, u_i1, v_i1 = 0.16, 0.08, 0.035, 0.16, 0.5, 0.25
Cons_u1, Cons_v1 = D_u_i1*(del_t/float(del_x**2)), D_v_i1*(del_t/float(del_x**2))

D_u_i2, D_v_i2, f_i2, k_i2, u_i2, v_i2 = 0.16, 0.08, 0.07, 0.32, 0.5, 0.25
Cons_u2, Cons_v2 = D_u_i2*(del_t/float(del_x**2)), D_v_i2*(del_t/float(del_x**2))

D_u_i3, D_v_i3, f_i3, k_i3, u_i3, v_i3 = 0.16, 0.08, 0.017, 0.08, 0.5, 0.25
Cons_u3, Cons_v3 = D_u_i3*(del_t/float(del_x**2)), D_v_i3*(del_t/float(del_x**2))

D_u_i4, D_v_i4, f_i4, k_i4, u_i4, v_i4 = 0.16, 0.08, 0.14, 0.64, 0.5, 0.25
Cons_u4, Cons_v4 = D_u_i4*(del_t/float(del_x**2)), D_v_i4*(del_t/float(del_x**2))

D_u_i5, D_v_i5, f_i5, k_i5, u_i5, v_i5 =  0.16, 0.08, 0.008, 0.04, 0.5, 0.25
Cons_u5, Cons_v5 = D_u_i5*(del_t/float(del_x**2)), D_v_i5*(del_t/float(del_x**2))

D_u_i6, D_v_i6, f_i6, k_i6, u_i6, v_i6 = 0.16, 0.08, 0.14, 0.16, 0.5, 0.25
Cons_u6, Cons_v6 = D_u_i6*(del_t/float(del_x**2)), D_v_i6*(del_t/float(del_x**2))

#f_i rate at which U is supplied
#k_i this plus f is rate at which V decays
#u_i initially everywhere in the system
#v_i intialy in a small square in the centre of the system 0 elsewhere

N = 100 #square size of grid
T = 10 #number of steps in interval 0->1

def fill_U(u):
    return np.full((N+1, N+1), u, dtype=float)

def fill_V(v):
    #create a square of chemical in the centre of the square
    V = np.zeros((N+1, N+1), dtype=float)
    for i in range(int(N/2 - 5), int(N/2 + 5)):
        for j in range(int(N/2 - 5), int(N/2 + 5)):
            V[i,j] = v
    return V

def fill_UV(u,v):
    for i in range(N):
        for j in range(N):
            if i >= (int(N/2 -2)) and i <= (int(N/2 + 2)):
                if j >= (int(N/2 - 2)) and j <= (int(N/2 + 2)):
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

def simulation(totT, mat_u, newu, u, Consu, mat_v, newv, v, Consv, f_i, k_i):
    for t in np.arange(totT):
        newu = Consu*(TD*mat_u+mat_u*TD + Boundary*mat_u)  \
        - (u * v**2 + (f_i * (1.0 - u)))
        newv = Consv*(TD*mat_v+mat_v*TD + Boundary*mat_v)  \
        + (u * v**2 - v * (f_i + k_i))
        mat_u = copy.deepcopy(newu)
        mat_v = copy.deepcopy(newv)
        u = np.array(newu)
        v = np.array(newv)
    return u, v


if __name__ == "__main__":
    #create range of matrices based on range of inputs
    for x in range(1,7):
        globals()["U%s" % x] = fill_U(globals()["u_i%s" % x])
        globals()["V%s" % x] = fill_V(globals()["v_i%s" % x])
        globals()["mat_U%s" % x] = np.matrix(globals()["U%s" % x])
        globals()["mat_V%s" % x] = np.matrix(globals()["V%s" % x])
        globals()["newU%s" % x] = copy.deepcopy(globals()["mat_U%s" % x])
        globals()["newV%s" % x] = copy.deepcopy(globals()["mat_V%s" % x])

    a = [1 for n in np.arange(N)]
    b = [0 for n in np.arange(N+1)]
    TD = tridiag(a,b,a)
    TD = np.matrix(TD)
    # Boundary[0,N] = 1
    # Boundary[N,0] = 1
    # Boundary[0,0] = 1
    # Boundary[N,N] = 1
    Boundary = np.zeros((N+1,N+1))
    Boundary = np.matrix(Boundary)
    print("Starting loop...")
    for x in range(1,7):
        globals()["newU%s" % x],globals()["newV%s" % x] = \
        simulation(T, globals()["mat_U%s" % x], \
        globals()["newU%s" % x], globals()["U%s" % x], \
        globals()["Cons_u%s" % x], globals()["mat_V%s" % x], \
        globals()["newV%s" % x], globals()["V%s" % x], globals()["Cons_v%s" % x],\
        globals()["f_i%s" % x], globals()["k_i%s" % x])
    start = time.time()
    end = time.time()
    print(end-start)

    fig, ax = plt.subplots(6,2)
    ax[0,0].matshow(newU1, cmap=plt.cm.gray)
    ax[0,1].matshow(newV1, cmap=plt.cm.gray)
    ax[1,0].matshow(newU2, cmap=plt.cm.gray)
    ax[1,1].matshow(newV2, cmap=plt.cm.gray)
    ax[2,0].matshow(newU3, cmap=plt.cm.gray)
    ax[2,1].matshow(newV3, cmap=plt.cm.gray)
    ax[3,0].matshow(newU4, cmap=plt.cm.gray)
    ax[3,1].matshow(newV4, cmap=plt.cm.gray)
    ax[4,0].matshow(newU5, cmap=plt.cm.gray)
    ax[4,1].matshow(newV5, cmap=plt.cm.gray)
    ax[5,0].matshow(newU6, cmap=plt.cm.gray)
    ax[5,1].matshow(newV6, cmap=plt.cm.gray)

    ax[0,0].set_title("U Concentration Grids")
    ax[0,1].set_title("V Concentration Grids")
    plt.show()
