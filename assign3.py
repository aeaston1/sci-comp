import numpy

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

def fill_UV(n_rows, n_cols, uv):
    return np.full((n_rows, n_cols), u, dtype=float)

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
