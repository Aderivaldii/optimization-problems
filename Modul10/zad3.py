import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import cvxpy as cp

"""
The script implements a solution to a linear programming (LP) problem using the Sequential Barrier Method (SBM).
The optimization problem is defined as:
    minimize    c^T x
    subject to  Ax ≤ b
This implementation includes an automatic method for finding a strictly feasible starting point, followed by 
the application of damped Newton's method to minimize the objective function augmented with a logarithmic barrier term.

Author: https://github.com/Pit1000
"""

# (65)
A = np.array([[0.4873, -0.8732],
              [0.6072, 0.7946],
              [0.9880, -0.1546],
              [-0.2142, -0.9768],
              [-0.9871, -0.1601],
              [0.9124, 0.4093]])
b = np.ones(6)
c = np.array([-0.5, 0.5])

# (69)
V = np.array([[0.1562, 0.9127, 1.0338, 0.8086, -1.3895, -0.8782],
              [-1.0580, -0.6358, 0.1386, 0.6406, 2.3203, -0.8311]])
V_closed = np.hstack([V, V[:, 0:1]])

# Funkcja znajdująca punkt ściśle dopuszczalny
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
def solve_lp_sbm(A, b, c, x0, t0=1.0, gamma=2.5, tol_outer=1e-3, newton_tol=1e-8, newton_max_iter=50):
    m, n = A.shape
    x = x0.copy()
    t = t0
    x_list = [x.copy()]

    def psi_t(x, t):
        if np.any(b - A.dot(x) <= 0):
            return np.inf
        return c.dot(x) - (1.0 / t) * np.sum(np.log(b - A.dot(x)))

    def grad_psi_t(x, t):
        residual = b - A.dot(x)
        return c + (1.0 / t) * A.T.dot(1.0 / residual)

    def hess_psi_t(x, t):
        residual = b - A.dot(x)
        weight = 1.0 / (residual ** 2)
        return (1.0 / t) * A.T.dot(weight[:, np.newaxis] * A)

    while m / t >= tol_outer:
        for i in range(newton_max_iter):
            g = grad_psi_t(x, t)
            H = hess_psi_t(x, t)
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("Macierz Hessiana osobliwa.")
                break
            lambda_sq = -g.dot(dx)
            if lambda_sq / 2.0 <= newton_tol:
                break
            alpha = 1.0
            tau = 0.01
            while True:
                x_new = x + alpha * dx
                if np.all(b - A.dot(x_new) > 0) and psi_t(x_new, t) <= psi_t(x, t) + tau * alpha * g.dot(dx):
                    break
                alpha *= 0.5
            x = x_new
        x_list.append(x.copy())
        t *= gamma
    return x, x_list

try:
    x_feas = find_strictly_feasible(A, b)
    print("Punkt ściśle dopuszczalny:", x_feas)
except ValueError as e:
    print(e)

x0 = x_feas
x_sbm, x_list = solve_lp_sbm(A, b, c, x0, t0=1.0, gamma=2.5, tol_outer=1e-3)
print("Rozwiązanie LP metodą SBM:", x_sbm)

res = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=[(None, None)] * len(c),
    method='highs',
    options={'maxiter': 1000} #'disp': True
)
if res.success:
    x_linprog = res.x
    print("Rozwiązanie LP metodą linprog:", x_linprog)
else:
    print("Linprog nie znalazł rozwiązania.")

# Wykresy
# rys 5
margin2 = 4
x_min2, x_max2 = V[0].min() - margin2, V[0].max() + margin2
y_min2, y_max2 = V[1].min() - margin2, V[1].max() + margin2
x_vals = np.linspace(x_min2, x_max2, 400)
plt.figure(figsize=(6,6))

for i in range(A.shape[0]):
    a1, a2 = A[i]
    y_line = (1 - a1 * x_vals) / a2
    plt.plot(x_vals, y_line, 'k--', linewidth=1)

plt.fill(V_closed[0, :], V_closed[1, :], color='grey', alpha=0.3)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim([-2.5, 2])
plt.ylim([-1.5, 3])
plt.grid(True)
plt.show()

# rys6
margin = 0.5
x_min_plot, x_max_plot = V[0].min() - margin, V[0].max() + margin
y_min_plot, y_max_plot = V[1].min() - margin, V[1].max() + margin

fig, ax = plt.subplots(figsize=(8, 6))
ax.fill(V_closed[0, :], V_closed[1, :], color="gray", alpha=0.3, label="Obszar dopuszczalny")
ax.plot(V_closed[0, :], V_closed[1, :], color="blue", linewidth=2)
ax.plot(x0[0], x0[1], 'ks', markersize=8, label="Punkt startowy")
x_list_arr = np.array(x_list)
ax.plot(x_list_arr[:, 0], x_list_arr[:, 1], 'ko', markersize=6, label="Punkty centralne (SBM)")
t_plot = 1.0
grid_x = np.linspace(x_min_plot, x_max_plot, 300)
grid_y = np.linspace(y_min_plot, y_max_plot, 300)
X, Y = np.meshgrid(grid_x, grid_y)
Z = np.full(X.shape, np.nan)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_val = X[i, j]
        y_val = Y[i, j]
        residuals = b - (A[:, 0] * x_val + A[:, 1] * y_val)
        if np.all(residuals > 0):
            Z[i, j] = c[0]*x_val + c[1]*y_val - (1.0/t_plot)*np.sum(np.log(residuals))

if np.any(~np.isnan(Z)):
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 10)
    cs = ax.contour(X, Y, Z, levels=levels, colors='m', linestyles='dashdot', linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

ax.plot(x_sbm[0], x_sbm[1], 'ro', markersize=8, label="Rozwiązanie SBM")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_xlim(x_min_plot, x_max_plot)
ax.set_ylim(y_min_plot, y_max_plot)
ax.legend()
ax.grid(True)
plt.show()

