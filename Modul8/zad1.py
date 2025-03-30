import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

'''
Nonlinear least squares problem with constraints
solved using the Augmented Lagrangian Algorithm (ALA)
with the Levenberg–Marquardt (LM) method. The code performs ALA iterations,
plots contour lines of ||f(x)||^2 and the curve g(x)=0 with marked iteration points.
Additionally, it computes FR (feasibility residual) and OR (optimality condition residual)
indices as functions of cumulative LM iterations.

Author: https://github.com/Pit1000
'''

def f_func(x):

    return np.array([
        x[0] + np.exp(-x[1]),
        x[0]**2 + 2.0*x[1] + 1.0
    ])

def g_func(x):

    return x[0] + x[0]**3 + x[1] + x[1]**2

def jac_f(x):

    return np.array([
        [1.0, -np.exp(-x[1])],
        [2.0*x[0], 2.0]
    ])

def jac_g(x):

    return np.array([
        [1.0 + 3.0*x[0]**2, 1.0 + 2.0*x[1]]
    ])

def residual(x, mu, z):

    res_constraint = np.sqrt(mu) * (g_func(x) + z/(2.0*mu))
    return np.concatenate([f_func(x), [res_constraint]])

def calc_FR(x):
    """
    residuum warunku dopuszczalnosci
    Feasibility residual = ||g(x)||_2
    """
    return abs(g_func(x))

def calc_OR(x, z):
    """
    residuum warunku optymalnosci
    OR (optimality residual) dla Augmented Lagrangian:
    """
    Jf = jac_f(x)          # (2,2)
    f_val = f_func(x)      # (2,)
    term1 = 2.0 * Jf.T.dot(f_val)  # (2,)
    Jg = jac_g(x)
    term2 = (Jg.T * z).ravel()

    return np.linalg.norm(term1 + term2)

# Parametry algorytmu ALA
x = np.array([0.5, -0.5])  # x(0)
mu = 1.0
z = 1.0
max_iter = 4

# Tablice historii
iterates = [x.copy()]
fr_vals = [calc_FR(x)]
or_vals = [calc_OR(x, z)]
cumulative_lm_iter_list = [0]  # Na początku 0
cumulative_lm_iter = 0

prev_g = abs(g_func(x))

# Pętla ALA z LM
for k in range(max_iter):
    sol = least_squares(residual, x, args=(mu, z), method='lm')
    x_new = sol.x

    n_lm_iter = sol.njev if sol.njev is not None else sol.nfev
    if n_lm_iter is None:
        n_lm_iter = 1
    cumulative_lm_iter += n_lm_iter

    z_new = z + 2.0*mu * (g_func(x_new) + z/(2.0*mu))

    g_norm_new = abs(g_func(x_new))
    if g_norm_new < 0.25 * prev_g:
        mu_new = mu
    else:
        mu_new = 2.0 * mu
    prev_g = g_norm_new

    x = x_new
    z = z_new
    mu = mu_new
    iterates.append(x.copy())

    fr_vals.append(calc_FR(x))
    or_vals.append(calc_OR(x, z))
    cumulative_lm_iter_list.append(cumulative_lm_iter)

#for i in iterates:
#    print(i)

print(x)

# Wykresy
# FR i OR
plt.figure(figsize=(7, 5))
plt.plot(cumulative_lm_iter_list, fr_vals, 'o-b', label='feasibility')
plt.plot(cumulative_lm_iter_list, or_vals, 's-r', label='opt. cond.')
plt.yscale('log')  # skala logarytmiczna osi Y
plt.xlabel('cumulative L-M iterations')
plt.ylabel('residual')
plt.grid(True, which='both')
plt.legend()
plt.title("FR i OR w funkcji skumulowanych iteracji Levenberga–Marquardta (ALA)")
#plt.show()

# Wykres
x1_vals = np.linspace(-2, 2, 200)
x2_vals = np.linspace(-2, 2, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

F = np.zeros_like(X1)
G = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        xx = np.array([X1[i, j], X2[i, j]])
        ff = f_func(xx)
        F[i, j] = np.sum(ff**2)  # ||f(x)||^2
        G[i, j] = g_func(xx)     # g(x)

plt.figure(figsize=(7, 6))

levels = [2, 6, 10, 20, 30]
contours_F = plt.contour(X1, X2, F, levels, colors='black')
plt.clabel(contours_F, inline=True, fontsize=8)

contours_g = plt.contour(X1, X2, G, levels=[0], colors='red', linewidths=2)

iterates = np.array(iterates)
plt.scatter(iterates[:, 0], iterates[:, 1], color='blue', label='Iteracje (ALA)')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend()
plt.grid(True)
plt.show()
