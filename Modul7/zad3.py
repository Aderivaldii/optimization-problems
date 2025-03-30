import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
First-order inertia: y(t) = k * (1 - exp(-t/T))
Parameters k and T are fitted using the Levenberg–Marquardt method
to measurement data (t, y) from the inertialData.mat file.

Author: https://github.com/Pit1000
'''

data = loadmat("inertialData.mat")
t = data['t'].squeeze()
y = data['y'].squeeze()

def model(x, t):
    k_, T_ = x
    return k_ * (1.0 - np.exp(-t / T_))


def residuals(x):
    return model(x, t) - y

def jacobian(x):
    k_, T_ = x
    m = len(t)
    J = np.zeros((m, 2))
    # df/dk
    J[:, 0] = 1.0 - np.exp(-t / T_)
    # df/dT
    J[:, 1] = -k_ * (t / T_ ** 2) * np.exp(-t / T_)
    return J

# Parametry początkowe i ustawienia algorytmu
x0 = np.array([1.0, 1.0])  # [k, T]
k_max = 30
lmbda = 1.0
n = len(x0)

# Zapis historii iteracji
lambda_hist = np.zeros(k_max + 1)
lambda_hist[0] = lmbda
obj_hist = np.zeros(k_max + 1)
obj_hist[0] = np.linalg.norm(residuals(x0)) ** 2

X = np.zeros((n, k_max + 1))
X[:, 0] = x0

# Główna pętla algorytmu Levenberga–Marquadta
for k in range(k_max):
    x = X[:, k]

    Jx = jacobian(x)
    fx = residuals(x)

    x_new = x - np.linalg.solve(
        Jx.T @ Jx + lmbda * np.eye(n),
        Jx.T @ residuals(x)
    )

    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(fx):
        X[:, k + 1] = x_new
        lmbda *= 0.8
    else:
        X[:, k + 1] = x
        lmbda *= 2.0

    lambda_hist[k + 1] = lmbda
    obj_hist[k + 1] = np.linalg.norm(residuals(X[:, k + 1])) ** 2


x_opt = X[:, -1]
print("Ostateczne parametry (k, T) =", x_opt)

# Wykresy
plt.figure()
plt.plot(t, y, 'o', label="measurement")
t_dense = np.linspace(t[0], t[-1], 300)
plt.plot(t_dense, model(x0, t_dense), label="first guess")
plt.plot(t_dense, model(x_opt, t_dense), label="final fit")
plt.title(r"$y = k \left[ 1 - e^{-t/T} \right]$")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

#Subplot
plt.figure(figsize=(6, 8))
# k
plt.subplot(2, 1, 1)
plt.plot(range(k_max + 1), X[0, :], 'k', linewidth=2)
plt.title("Parametr k od iteracji")
plt.xlabel("k - iteracja")
plt.ylabel("k(k {iteracja})")
plt.grid(True)

# T
plt.subplot(2, 1, 2)
plt.plot(range(k_max + 1), X[1, :], 'k', linewidth=2)
plt.title("Parametr T od iteracji")
plt.xlabel("k - iteracja")
plt.ylabel("T(k)")
plt.grid(True)
plt.tight_layout()
plt.show()

# lambda
plt.figure()
plt.scatter(range(k_max + 1), lambda_hist, marker='s', color='black')
plt.title("Parametr lambda od iteracji")
plt.xlabel("k")
plt.ylabel(r"$\lambda(k)$")
plt.grid(True)
plt.show()

# ‖f(x)‖²
plt.figure()
plt.scatter(range(k_max + 1), obj_hist, color = 'black', marker = 's')
plt.title("Wartość funkcji celu od iteracji")
plt.xlabel("k")
plt.ylabel("‖f(x)‖²")
plt.grid(True)
plt.show()



