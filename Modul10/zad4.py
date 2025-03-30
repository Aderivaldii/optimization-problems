import numpy as np
from scipy.optimize import linprog
import cvxpy as cp

"""
This is an attempt to implement a solution to a Linear Programming (LP) problem using the Sequential Barrier Method (SBM)
for arbitrary matrices.

Author: https://github.com/Pit1000
"""

def find_strictly_feasible(A, b, tol=1e-6):

    m, n = A.shape
    x = cp.Variable(n)
    s = cp.Variable()

    constraints = [A @ x - b <= s]

    objective = cp.Minimize(s)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status in ["infeasible", "unbounded"]:
        raise ValueError("Zadanie pomocnicze jest niewykonalne.")

    if s.value < -tol:
        return x.value
    else:
        raise ValueError("Brak ściśle dopuszczalnego punktu startowego (s* >= 0).")

# Metoda sekwencyjnej bariery (SBM)
def solve_lp_sbm_general(A, b, c, t0=1.0, gamma=2.5, tol_outer=1e-3,
                         newton_tol=1e-8, newton_max_iter=50):

    m, n = A.shape
    x0 = find_strictly_feasible(A, b)

    x = x0.copy()
    t = t0
    x_list = [x.copy()]

    # Funkcja bariery
    def psi_t(x, t):
        residual = b - A.dot(x)
        if np.any(residual <= 0):
            return np.inf
        return c.dot(x) - (1.0 / t) * np.sum(np.log(residual))

    def grad_psi_t(x, t):
        residual = b - A.dot(x)
        return c + (1.0 / t) * A.T.dot(1.0 / residual)

    def hess_psi_t(x, t):
        residual = b - A.dot(x)
        weight = 1.0 / (residual ** 2)
        return (1.0 / t) * A.T.dot(np.diag(weight)).dot(A)

    while m / t >= tol_outer:
        for i in range(newton_max_iter):
            g = grad_psi_t(x, t)
            H = hess_psi_t(x, t)
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("Macierz Hessiana osobliwa. Przerywamy iteracje Newtona.")
                break
            lambda_sq = -g.dot(dx)
            if lambda_sq / 2.0 <= newton_tol:
                break
            # Backtracking line search
            alpha = 1.0
            tau = 0.01
            while True:
                x_new = x + alpha * dx
                if np.all(b - A.dot(x_new) > 0) and \
                        psi_t(x_new, t) <= psi_t(x, t) + tau * alpha * g.dot(dx):
                    break
                alpha *= 0.5
            x = x_new
        x_list.append(x.copy())
        t *= gamma
    return x, x_list

A = np.array([
    [-1, 0],
    [0, -1],
    [1, 1]
])
b = np.array([-1, -2, 5])
c = np.array([1, -1])

# SBM
x_solution, x_iterates = solve_lp_sbm_general(A, b, c)
#print("Punkt ściśle dopuszczalny:", x_iterates[0])
print("Rozwiązanie LP metodą SBM:", x_solution)

# linprog
res = linprog(c, A_ub=A, b_ub=b, method='highs')
if res.success:
    print("Rozwiązanie LP metodą linprog:", res.x)
else:
    print("Linprog nie znalazł rozwiązania.")
