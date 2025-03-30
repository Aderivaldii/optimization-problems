import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
Double inertia: y(t) = k * [ 1 - (1/(T1 - T2)) * (T1*exp(-t/T1) - T2*exp(-t/T2)) ]
Parameters k, T1, T2 are fitted to measurement data (t, y) from inertialData.mat
using the Levenberg–Marquardt method.

Author: https://github.com/Pit1000
'''

# Wczytanie danych z pliku (zakładamy, że w pliku mamy tablice t i y)
data = loadmat("twoInertialData.mat")
t = data['t'].squeeze()
y = data['y'].squeeze()

def model(x, t):
    k_, T1_, T2_ = x
    return k_ * (1.0 - 1.0/(T1_ - T2_) * (T1_ * np.exp(-t/T1_) - T2_ * np.exp(-t/T2_)))

def residuals(x):
    return model(x, t) - y

def jacobian(x):
    k_, T1_, T2_ = x
    m = len(t)
    J = np.zeros((m, 3))
    exp1 = np.exp(-t/T1_)
    exp2 = np.exp(-t/T2_)
    # df/dk
    J[:, 0] = 1.0 - 1.0/(T1_ - T2_) * (T1_*exp1 - T2_*exp2)
    # df/T1
    J[:, 1] = (k_/(T2_ - T1_) * ((t / T1_) * exp1 + (T2_ / (T1_ - T2_)) * (exp2 - exp1)))
    # df/T2
    J[:, 2] = (k_/(T1_ - T2_) * ((t / T2_) * exp2 + (T1_ / (T2_ - T1_)) * (exp1 - exp2)))
    return J

# Parametry początkowe i ustawienia algorytmu
x0 = np.array([1.0, 1.0, 2.0]) #[k, T1, T2]
k_max = 30
lmbda = 1.0
n = len(x0)

# Zapis historii iteracji
lambda_hist = np.zeros(k_max + 1)
lambda_hist[0] = lmbda
obj_hist = np.zeros(k_max + 1)
obj_hist[0] = np.linalg.norm(residuals(x0))**2

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
k_opt, T1_opt, T2_opt = x_opt
print("Ostateczne parametry (k, T1, T2) =", x_opt)

# Wykresy
plt.figure()
plt.plot(t, y, 'o', label="measurement")
t_dense = np.linspace(t[0], t[-1], 400)
plt.plot(t_dense, model(x0, t_dense), label="first guess")
plt.plot(t_dense, model(x_opt, t_dense), label="final fit")
plt.title(r"$y(t) = k \left[ 1 - \frac{1}{T_1 - T_2} \left(T_1 e^{-t/T_1} - T_2 e^{-t/T_2}\right)\right]$")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(6, 8))
# k
plt.subplot(3, 1, 1)
plt.plot(range(k_max+1), X[0, :], 'k', linewidth=2)
plt.title("Parametr k od iteracji")
plt.xlabel("k - iteracja")
plt.ylabel("k")
plt.grid(True)
# T1
plt.subplot(3, 1, 2)
plt.plot(range(k_max+1), X[1, :], 'k', linewidth=2)
plt.title(r"Parametr $T_1$ od iteracji")
plt.xlabel("k - iteracja")
plt.ylabel(r"$T_1$")
plt.grid(True)
# T2
plt.subplot(3, 1, 3)
plt.plot(range(k_max+1), X[2, :], 'k', linewidth=2)
plt.title(r"Parametr $T_2$ od iteracji")
plt.xlabel("k - iteracja")
plt.ylabel(r"$T_2$")
plt.grid(True)
plt.tight_layout()
plt.show()

# lambda
plt.figure()
plt.scatter(range(k_max + 1), lambda_hist, marker='s', color='black')
plt.title("Parametr lambda od iteracji")
plt.xlabel("k iteracja")
plt.ylabel(r"$\lambda(k)$")
plt.grid(True)
plt.show()

# ‖f(x)‖²
plt.figure()
plt.scatter(range(k_max + 1), obj_hist, color='black', marker='s')
plt.title("Wartość funkcji celu od iteracji")
plt.xlabel("k iteracja")
plt.ylabel("‖f(x)‖²")
plt.grid(True)
plt.show()
