import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from itertools import product

'''
Solves the Boolean Least Squares problem, which consists in minimizing the objective function ||Ax - b||²,
subject to the constraint that each element of vector x takes the value -1 or +1 (x_i² = 1 for i = 1, …, n).

This script demonstrates both an exhaustive (brute force) approach and an efficient solution 
to the Boolean Least Squares problem using the ALA algorithm, which iteratively minimizes 
the objective function under imposed nonlinear constraints.

Author: https://github.com/Pit1000
'''

def boolean_brute_force(A, b):
    m, n = A.shape
    best_obj = np.inf
    best_x = None
    for candidate in product([-1, 1], repeat=n):
        x_candidate = np.array(candidate)
        obj = np.linalg.norm(A.dot(x_candidate) - b) ** 2
        if obj < best_obj:
            best_obj = obj
            best_x = x_candidate.copy()
    return best_x, best_obj

def boolean_residual(x, A, b, mu, z):
    r1 = A.dot(x) - b
    g = x ** 2 - 1
    r2 = np.sqrt(mu) * (g + z / (2.0 * mu))
    return np.concatenate([r1, r2])

def boolean_jacobian(x, A, b, mu, z):
    m, n = A.shape
    J1 = A
    J2 = np.diag(2.0 * x) * np.sqrt(mu)
    return np.vstack([J1, J2])

def boolean_ALA(A, b, x0, max_iter=50, tol=1e-5):

    m, n = A.shape
    z = np.zeros(n)
    mu = 1.0
    x = x0.copy()

    best_obj = np.inf
    best_x = None
    obj_hist = []

    for k in range(max_iter):
        sol = least_squares(boolean_residual, x, args=(A, b, mu, z), method='lm')
        x = sol.x

        g_val = x ** 2 - 1

        z = z + 2.0 * mu * (g_val + z / (2.0 * mu))

        if np.linalg.norm(g_val, 2) < 0.25 * np.linalg.norm(g_val, 2):
            mu = mu
        else:
            mu = 2.0 * mu

        x_round = np.sign(x)
        obj = np.linalg.norm(A.dot(x_round) - b) ** 2
        obj_hist.append(obj)

        if obj < best_obj:
            best_obj = obj
            best_x = x_round.copy()

        if np.linalg.norm(g_val, 2) < tol:
            break

    return best_x, best_obj, obj_hist

# parametry
np.random.seed(457)
m = 10
n = 10
A = np.random.randn(m, n)
b = np.random.randn(m)

bf_solution, bf_obj = boolean_brute_force(A, b)
print("Brute force - najlepsze rozwiązanie:")
print(bf_solution)
print("Wartość celu ||Ax - b||^2:", bf_obj)

x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

best_x, best_obj, obj_hist = boolean_ALA(A, b, x0, max_iter=50, tol=1e-5)

print("Najlepsze Boolean rozwiązanie:", best_x)
print("Wartość celu ||Ax - b||^2:", best_obj)

# plt.figure(figsize=(6, 4))
# plt.plot(obj_hist, marker='o')
# plt.xlabel("Iteracja (ALA)")
# plt.ylabel("||Ax̃ - b||²")
# plt.grid(True)
# plt.show()
