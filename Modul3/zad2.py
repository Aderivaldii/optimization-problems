import cvxpy as cp
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

"""
This script performs reconstruction of a piecewise constant signal using an approach based on linear programming (LP).
The original problem is reformulated into an LP form, allowing it to be solved by minimizing the L1 norm of the measurement error and the sum of absolute differences of the signal. 
This method provides an alternative to the classical LASSO formulation and employs convex optimization techniques.

Author: https://github.com/Pit1000
"""

data = scipy.io.loadmat("Data01.mat")
T = np.array(data["t"]).flatten()
Y = np.array(data["y"]).flatten()
n = len(Y)

D = np.eye(n-1, n, k=1) - np.eye(n-1, n)
m = D.shape[0]

# Parametry
q = 1.5
tau = 10.0

#q

v_1 = cp.Variable(n)
xi_1 = cp.Variable(n, nonneg=True)
delta_1 = cp.Variable(m, nonneg=True)

constraints_1 = []
for i in range(n):
    ## (20b)
    constraints_1 += [
        (Y[i] - v_1[i] <= xi_1[i]),
        (-(Y[i] - v_1[i]) <= xi_1[i])
    ]

Dv_5b = D @ v_1
for j in range(m):
    ## (20d)
    constraints_1 += [
        (Dv_5b[j] <= delta_1[j]),
        (-Dv_5b[j] <= delta_1[j])
    ]

## 20c
constraints_1.append(cp.sum(delta_1) <= q)

objective_1 = cp.Minimize(cp.sum(xi_1))

problem_5b = cp.Problem(objective_1, constraints_1)
problem_5b.solve()

v_opt_1 = v_1.value

#tau

v_2 = cp.Variable(n)
xi_2 = cp.Variable(n, nonneg=True)
delta_2 = cp.Variable(m, nonneg=True)

#28b
constraints_2 = []
for i in range(n):
    constraints_2 += [
        (Y[i] - v_2[i] <= xi_2[i]),
        (-(Y[i] - v_2[i]) <= xi_2[i])
    ]
#28c
Dv_2 = D @ v_2
for j in range(m):
    constraints_2 += [
        (Dv_2[j] <= delta_2[j]),
        (-Dv_2[j] <= delta_2[j])
    ]

#28a
objective_2 = cp.Minimize(cp.sum(xi_2) + tau * cp.sum(delta_2))
problem_2 = cp.Problem(objective_2, constraints_2)
problem_2.solve()

v_opt_2 = v_2.value

#Wizualizacja wyników
plt.figure(figsize=(10, 5))
plt.plot(T, Y, label="Oryginalny sygnał", color="black", marker='.', linestyle='none')
plt.plot(T, v_opt_1, label="LP - 1. (q = " + str(q) + ")")
plt.plot(T, v_opt_2, label="LP - 2. (tau = " + str(tau) + ")")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.title("Rekonstrukcja sygnału kawałkami stałego LASSO - LP")
plt.show()
