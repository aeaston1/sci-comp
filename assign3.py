import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import copy
import time

del_t = 1
del_x = 1
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
T = 300 #number of steps in interval 0->1

def fill_U(u):
    #fill the array with u value everywhere
    return np.full((N+1, N+1), u, dtype=float)

def fill_V(v):
    #create a square of chemical in the centre of the square
    V = np.zeros((N+1, N+1), dtype=float)
    for i in range(int(N/2 - 5), int(N/2 + 5)):
        for j in range(int(N/2 - 5), int(N/2 + 5)):
            V[i,j] = v
    return V

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def simulation(totT, mat_u, newu, u, Consu, mat_v, newv, v, Consv, f_i, k_i):
    for t in np.arange(totT):
        #Gray-Scott equations
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

    #creation of tridiagonal matrices
    a = [1 for n in np.arange(N)]
    b = [0 for n in np.arange(N+1)]
    TD = tridiag(a,b,a)
    TD = np.matrix(TD)
    # boundaries
    Boundary = np.zeros((N+1,N+1))
    Boundary[0,N] = 1
    Boundary[N,0] = 1
    Boundary[0,0] = 1
    Boundary[N,N] = 1
    Boundary = np.matrix(Boundary)
    print("Starting loop...")
    # running multiple simulations for various parameters
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

    fig, ax = plt.subplots(3,2)
    fig1, ax1 = plt.subplots(3,2)
    ax00 = ax[0,0].matshow(newU1, cmap=plt.cm.gray)
    ax01 = ax[0,1].matshow(newV1, cmap=plt.cm.gray)
    ax10 = ax[1,0].matshow(newU2, cmap=plt.cm.gray)
    ax11 = ax[1,1].matshow(newV2, cmap=plt.cm.gray)
    ax20 = ax[2,0].matshow(newU3, cmap=plt.cm.gray)
    ax21 = ax[2,1].matshow(newV3, cmap=plt.cm.gray)
    ax100 = ax1[0,0].matshow(newU4, cmap=plt.cm.gray)
    ax101 = ax1[0,1].matshow(newV4, cmap=plt.cm.gray)
    ax110 = ax1[1,0].matshow(newU5, cmap=plt.cm.gray)
    ax111 = ax1[1,1].matshow(newV5, cmap=plt.cm.gray)
    ax120 = ax1[2,0].matshow(newU6, cmap=plt.cm.gray)
    ax121 = ax1[2,1].matshow(newV6, cmap=plt.cm.gray)

    #colorbars
    fig.colorbar(ax00, ax=ax[0,0])
    fig.colorbar(ax01, ax=ax[0,1])
    fig.colorbar(ax10, ax=ax[1,0])
    fig.colorbar(ax11, ax=ax[1,1])
    fig.colorbar(ax20, ax=ax[2,0])
    fig.colorbar(ax21, ax=ax[2,1])
    fig.colorbar(ax100, ax=ax1[0,0])
    fig.colorbar(ax101, ax=ax1[0,1])
    fig.colorbar(ax110, ax=ax1[1,0])
    fig.colorbar(ax111, ax=ax1[1,1])
    fig.colorbar(ax120, ax=ax1[2,0])
    fig.colorbar(ax121, ax=ax1[2,1])

    ax[0,0].set_title("U - (a)", fontsize=20)
    ax[0,1].set_title("V - (a)",  fontsize=20)
    ax[1,0].set_title("U - (b)", fontsize=20)
    ax[1,1].set_title("V - (b)",  fontsize=20)
    ax[2,0].set_title("U - (c)", fontsize=20)
    ax[2,1].set_title("V - (c)",  fontsize=20)
    ax1[0,0].set_title("U - (d)",  fontsize=20)
    ax1[0,1].set_title("V - (d)",  fontsize=20)
    ax1[1,0].set_title("U - (e)",  fontsize=20)
    ax1[1,1].set_title("V - (e)",  fontsize=20)
    ax1[2,0].set_title("U - (f)",  fontsize=20)
    ax1[2,1].set_title("V - (f)",  fontsize=20)
    plt.show()
