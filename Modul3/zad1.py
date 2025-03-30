import cvxpy as cp
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

"""
The code performs reconstruction of a piecewise constant signal that has been recorded with the presence of noise. 
The problem is to estimate the original signal by minimizing the measurement error (L2 norm), while simultaneously limiting 
the total variation of the signal, achieved by applying the L1 norm on the differences between consecutive samples. 
This approach belongs to convex optimization methods, specifically employing the LASSO (Least Absolute Shrinkage and Selection Operator) heuristic.

Author: https://github.com/Pit1000
"""

data = scipy.io.loadmat("Data01.mat")
T = np.array(data["t"]).flatten()
Y = np.array(data["y"]).flatten()

n = len(Y)

D = np.eye(n-1, n, k=1) - np.eye(n-1, n)

v = cp.Variable(n)

# Parametr
q = 1.5
tau = 10.0

# 5ab
objective = cp.Minimize(cp.norm2(Y - v))
constraints = [cp.norm1(D @ v) <= q]
problem = cp.Problem(objective, constraints)

# 6
#objective = cp.Minimize(cp.norm2(Y - v) + tau * cp.norm1(D @ v))
#problem = cp.Problem(objective)

problem.solve()

v_opt = v.value

# Wizualizacja wyników
plt.figure(figsize=(10, 5))
plt.plot(T, Y, label="Oryginalny sygnał", color="black", marker='.', linestyle='none')
plt.plot(T, v_opt, label="LASSO", color="red")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.title("Rekonstrukcja sygnału kawałkami stałego LASSO")
plt.show()
