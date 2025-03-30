import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
In this code, we fit a damped sinusoid of the form:
    y(t) = A * e^(-a * t) * sin(ω * t + φ)
to the measurement data (t, y) from the file LM04Data.mat.
The Levenberg–Marquardt method is used to minimize
the sum of squared differences between the model and data values.
The parameters being estimated are: A (amplitude), 
a (damping coefficient), ω (angular frequency), and φ (phase shift).

Author: https://github.com/Pit1000
'''

data = loadmat("LM04Data.mat")
t = data['t'].squeeze()
y = data['y'].squeeze()

def model(x, t):
    A, a, w, phi = x
    return A * np.exp(-a * t) * np.sin(w * t + phi)

def residuals(x):
    return model(x, t) - y

def jacobian(x):
    A, a, w, phi = x
    m = len(t)
    J = np.zeros((m, 4))
    # df/dA
    J[:, 0] = np.exp(-a * t) * np.sin(w * t + phi)
    # df/da
    J[:, 1] = -t * A * np.exp(-a * t) * np.sin(w * t + phi)
    # df/domega
    J[:, 2] = A * np.exp(-a * t) * t * np.cos(w * t + phi)
    # df/dphi
    J[:, 3] = A * np.exp(-a * t) * np.cos(w * t + phi)
    return J

# Parametry początkowe i ustawienia algorytmu
x0 = np.array([1.0, 1.0, 50.0, 0.0])  # [A, a, omega, phi]
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

    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(residuals(x)):
        X[:, k + 1] = x_new
        lmbda *= 0.8
    else:
        X[:, k + 1] = x
        lmbda *= 2.0

    lambda_hist[k + 1] = lmbda
    obj_hist[k + 1] = np.linalg.norm(residuals(X[:, k + 1])) ** 2

x_opt = X[:, -1]

print("Ostateczne parametry (A, a, omega, phi) =", x_opt)

#Wykresy
plt.figure()
plt.plot(t, y, 'o', label="measurement")  # dane pomiarowe

t_dense = np.linspace(t[0], t[-1], 1000)
plt.plot(t_dense, model(x0, t_dense), label="first guess")  # aproksymacja startowa
plt.plot(t_dense, model(x_opt, t_dense), label="final fit")  # aproksymacja końcowa

plt.title(r'Dopasowanie $y = A e^{-a t} sin(ω t + φ)$')
plt.xlabel("t")
plt.ylabel(r'$y = A e^{-a t} sin(ω t + φ)$')
plt.grid(True)
plt.legend()
plt.show()

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
plt.xlim(-0.5, 26)
plt.grid(True)
plt.show()

# ‖f(x)‖²
plt.figure()
plt.scatter(range(k_max + 1), obj_hist, color = 'black', marker = 's')
plt.title("Wartość funkcji celu od iteracji")
plt.xlabel("k")
plt.ylabel("‖f(x)‖²")
plt.xlim(-0.5, 26)
plt.grid(True)
plt.show()
