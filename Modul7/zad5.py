import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

'''
The goal is to determine parameters (k, γ, β) for a second-order system
with transfer function:
      Gosc(s) = k / (1 + 2ξTs + T²s²)
such that its step response closely matches the step response of a target
fourth-order system obtained empirically or through simulation.

Author: https://github.com/Pit1000
'''

data = loadmat("reductionData.mat")
t = data['t'].squeeze()
y = data['y'].squeeze()

t_scaled = 1e3 * t
y_scaled = 1e3 * y

def model(x, t):
    k_, gamma_, beta_ = x
    term_exp = np.exp(-gamma_ * t)
    return k_ * (1.0 - term_exp * (np.cos(beta_ * t) + (gamma_ / beta_) * np.sin(beta_ * t)))

def residuals(x):
    return model(x, t_scaled) - y_scaled

def jacobian(x):
    k_, gamma_, beta_ = x
    N = len(t_scaled)
    J = np.zeros((N, 3))

    term_exp = np.exp(-gamma_ * t_scaled)
    cos_part = np.cos(beta_ * t_scaled)
    sin_part = np.sin(beta_ * t_scaled)

    # df/dk
    J[:, 0] = 1.0 - term_exp * (cos_part + (gamma_ / beta_) * sin_part)
    # df/dgamma
    J[:, 1] = k_ * term_exp * (t_scaled * cos_part - ((1.0 - gamma_ * t_scaled) / beta_) * sin_part)
    # df/dbeta
    J[:, 2] = k_ * term_exp * ((t_scaled + (gamma_ / (beta_ ** 2))) * sin_part - (gamma_ / beta_) * t_scaled * cos_part)
    return J

#incjalizacja algorytmu
x0 = np.array([1.0, 1.0, 1.0])  # punkt startowy dla problemu skalowanego
k_max = 25
lmbda = 1.0

X = np.zeros((3, k_max + 1))
X[:, 0] = x0
lambda_hist = np.zeros(k_max + 1)
lambda_hist[0] = lmbda
obj_hist = np.zeros(k_max + 1)
obj_hist[0] = np.linalg.norm(residuals(x0)) ** 2

# Główna pętla algorytmu Levenberga–Marquadta
for k in range(k_max):
    x = X[:, k]

    Jx = jacobian(x)
    fx = residuals(x)

    x_new = x - np.linalg.solve(
        Jx.T @ Jx + lmbda * np.eye(3),
        Jx.T @ residuals(x)
    )

    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(fx):
        X[:, k + 1] = x_new
        lmbda *= 0.8
    else:
        X[:, k + 1] = x
        lmbda *= 2.0

    lambda_hist[k + 1] = lmbda
    obj_hist[k + 1] = np.linalg.norm(residuals(X[:, k + 1]))**2

x_opt_scaled = X[:, -1]
x_opt_true = x_opt_scaled * 1e-3

print("Ostateczne parametry (dla skalowanych danych):", x_opt_scaled)
print("Ostateczne parametry (po przeskalowaniu):", x_opt_true)
print("Wartość funkcji celu =", obj_hist[-1])

# Wyniki
plt.figure()
plt.plot(t_scaled, y_scaled, color = 'orange', label="measurement", linewidth=3)
t_dense_scaled = np.linspace(t_scaled[0], t_scaled[-1], 1000)
plt.plot(t_dense_scaled, model(x0, t_dense_scaled), color = 'yellow', label="first guess")
plt.plot(t_dense_scaled, model(x_opt_scaled, t_dense_scaled), '--', color = 'blue', label=f"final fit (k = {k_max})", linewidth=1)
plt.title("Dopasowanie 2-rzędowego układu oscylacyjnego")
plt.xlabel(r"$t$ (skalowane)")
plt.ylabel(r"$h, hosc$ (skalowane)")
plt.grid(True)
plt.legend()
plt.show()

# k, gamma, beta
plt.figure()
plt.plot(X[0, :], color = 'black', label='k (skalowane)', linewidth=2)
plt.plot(X[1, :], color = 'red', label='gamma (skalowane)', linewidth=2)
plt.plot(X[2, :], color = 'blue', label='beta (skalowane)', linewidth=2)
plt.title("Zmiana parametrów k, gamma, beta od iteracji")
plt.xlabel("k iteracje")
plt.ylabel(r"$k, \gamma, \beta$")
plt.grid(True)
plt.legend()
plt.show()


# ‖f(x)‖²
plt.figure()
plt.plot(range(k_max + 1), obj_hist, 'r-o')
plt.title("Wartość funkcji celu ‖f(x)‖² w kolejnych iteracjach")
plt.xlabel("k iteracje")
plt.ylabel("‖f(x)‖²")
plt.grid(True)
plt.show()

# lambda
plt.figure()
plt.plot(range(k_max + 1), lambda_hist, 'b-o')
plt.title("Parametr lambda w kolejnych iteracjach")
plt.xlabel("k iteracje")
plt.ylabel(r"$\lambda$")
plt.grid(True)
plt.show()

