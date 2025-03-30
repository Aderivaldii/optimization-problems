import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
In this code, we fit a sinusoidal function of the form:
    y = A * sin(ω * t + φ)
to experimental data points (t, y) loaded from the file LM01Data.mat.
Levenberg–Marquardt algorithm was applied to minimize the sum of squared differences between the model and the data.
The parameters being estimated are: A (amplitude), ω (angular frequency), and φ (phase shift).

Author: https://github.com/Pit1000
'''

data = loadmat("LM01Data.mat")
t = data['t'].squeeze()
y = data['y'].squeeze()

# Parametry początkowe i ustawienia algorytmu
x0 = np.array([1.0, 100.0 * np.pi, 0.0])  # [A, omega, phi]
k_max = 35
lmbda = 1.0
n = len(x0)

def model(x, t):
    return x[0] * np.sin(x[1] * t + x[2])

def residuals(x):
    return model(x, t) - y

def jacobian(x):
    A, omega, phi = x
    m = len(t)
    J = np.zeros((m, n))
    # df/dA
    J[:, 0] = np.sin(omega * t + phi)
    # df/domega
    J[:, 1] = A * t * np.cos(omega * t + phi)
    # df/dphi
    J[:, 2] = A * np.cos(omega * t + phi)
    return J

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

    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(residuals(x)):
        X[:, k + 1] = x_new
        lmbda *= 0.8
    else:
        X[:, k + 1] = x
        lmbda *= 2.0

    lambda_hist[k + 1] = lmbda
    obj_hist[k + 1] = np.linalg.norm(residuals(X[:, k + 1])) ** 2

x_opt = X[:, -1]

print("Wynik końcowy (A, omega, phi) =", x_opt)

# Wykresy
plt.figure()
plt.plot(t, y, 'o', label='measurement')
t_dense = np.linspace(t[0], t[-1], 1000)
plt.plot(t_dense, model(x0, t_dense), label='first guess')
plt.plot(t_dense, model(x_opt, t_dense), label='final fit')
plt.legend()
plt.title("Dopasowanie sinusoidy metodą LM")
plt.xlabel("t [a.u.]")
plt.ylabel(r'$y(t) = Asin(ωt + \phi)$ [a.u.]')
plt.grid(True)
plt.show()

# # A
# plt.figure()
# plt.plot(range(k_max + 1), X[0, :], label='A')
# plt.title("Parametr A od iteracji")
# plt.xlabel("k")
# plt.ylabel("A(k)")
# plt.grid(True)
# plt.show()
#
# # omega
# plt.figure()
# plt.plot(range(k_max + 1), X[1, :], label='omega')
# plt.title("Parametr omega od iteracji")
# plt.xlabel("k")
# plt.ylabel("ω(k)")
# plt.grid(True)
# plt.show()
#
# # phi
# plt.figure()
# plt.plot(range(k_max + 1), X[2, :], label='phi')
# plt.title(r'Parametr $\phi(k)$ od iteracji')
# plt.xlabel("k")
# plt.ylabel(r'$\phi(k)$')
# plt.grid(True)
# plt.show()

#Subplot
plt.figure(figsize=(6, 8))
# A
plt.subplot(3, 1, 1)
plt.plot(range(k_max + 1), X[0, :], 'k', linewidth=2)
plt.title("Parametr A od iteracji")
plt.xlabel("k")
plt.ylabel("A(k)")
plt.grid(True)

# omega
plt.subplot(3, 1, 2)
plt.plot(range(k_max + 1), X[1, :], 'k', linewidth=2)
plt.title("Parametr omega od iteracji")
plt.xlabel("k")
plt.ylabel("ω(k)")
plt.grid(True)

# phi
plt.subplot(3, 1, 3)
plt.plot(range(k_max + 1), X[2, :], 'k', linewidth=2)
plt.title(r'Parametr $\phi(k)$ od iteracji')
plt.xlabel("k")
plt.ylabel(r'$\phi(k)$')
plt.grid(True)
plt.tight_layout()
plt.show()

# lambda
plt.figure()
plt.scatter(range(k_max + 1), lambda_hist, color = 'black', marker = 's')
plt.title("Parametr lambda od iteracji")
plt.xlabel("k")
plt.ylabel("λ(k)")
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
