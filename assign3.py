import numpy as np
import matplotlib.pyplot as plt

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

def fill_U(u):
    return np.full((n_rows_i, n_cols_i), u, dtype=float)

def fill_V(v):
    #create a square of chemical in the centre of the square
    V = np.zeros((n_rows_i, n_cols_i), dtype=float)
    for i in range(int(n_rows_i/2 - 2), int(n_rows_i/2 + 2)):
        for j in range(int(n_cols_i/2 - 2), int(n_cols_i/2 + 2)):
            V[i,j] = v
    return V

def u_to_v(u, v):
    new_v = u + 2*v
    return 3 * new_v

def v_to_p(v, p):
    p = copy.deepcopy(v)
    return p

def u_time(d_t, D_u, u, v, f, i,j):
    return d_t * (D_u * u[i,j]) - (u[i,j] * (v[i,j])**2) + (f * (1-u[i,j]))

def v_time(d_t, D_v, u, v, f, k, i, j):
    return d_t * (D_v * v[i,j]) + (u[i,j] * (v[i,j])**2) - (v * (f + k))


if __name__ == "__main__":
    U = fill_U(u_i)
    V = fill_V(v_i)
    plt.imshow(V)
    plt.show()
