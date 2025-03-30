import numpy as np

"""
Script for finding the minimum of a function using quasi-Newton methods
with updates:

    - SR1 (Symmetric Rank One)
    - DFP (Davidon-Fletcher-Powell)
    - BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Author: https://github.com/Pit1000
"""

def f(x):
    x1, x2 = x
    term1 = np.exp(x1 + 3 * x2 - 0.1)
    term2 = np.exp(-x1 - 0.1)
    xc = np.array([1.0, 1.0])
    P = (1 / 8) * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])
    term3 = np.dot(x - xc, P.dot(x - xc))
    return term1 + term2 + term3

def grad_f(x):
    x1, x2 = x
    term1 = np.exp(x1 + 3 * x2 - 0.1)
    term2 = np.exp(-x1 - 0.1)
    grad1 = term1 - term2
    grad2 = 3 * term1
    xc = np.array([1.0, 1.0])
    P = (1 / 8) * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])
    grad_quad = 2 * P.dot(x - xc)
    return np.array([grad1, grad2]) + grad_quad

def quasi_newton(f, grad_f, x0, tol=1e-4, alpha=0.5, beta=0.5, method='SR1', max_iter=1000):
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    iterations = 0
    while iterations < max_iter:
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        v = -H.dot(g)
        xi = -np.dot(g, v)
        if xi < tol:
            break
        s = 1.0
        while f(x + s * v) > f(x) + s * alpha * np.dot(g, v):
            s *= beta
        x_new = x + s * v
        Δx = x_new - x
        g_new = grad_f(x_new)
        Δg = g_new - g
        if np.dot(Δg, Δx) == 0:
            break

        if method == 'SR1':
            u = Δx - H.dot(Δg)
            denominator = np.dot(u, Δg)
            if abs(denominator) > 1e-8:
                H = H + np.outer(u, u) / denominator
        elif method == 'DFP':
            term1 = np.outer(Δx, Δx) / np.dot(Δg, Δx)
            temp = H.dot(Δg)
            term2 = np.outer(temp, temp) / np.dot(Δg, temp)
            H = H + term1 - term2
        elif method == 'BFGS':
            rho = 1.0 / np.dot(Δg, Δx)
            I = np.eye(n)
            H = (I - rho * np.outer(Δx, Δg)).dot(H).dot(I - rho * np.outer(Δg, Δx)) + rho * np.outer(Δx, Δx)
        else:
            raise ValueError("Nieznana metoda aktualizacji macierzy H.")
        x = x_new
        iterations += 1
    return x, f(x), iterations


x0 = [2, -2]
tol = 1e-4
alpha = 0.5
beta = 0.5
methods = ['SR1', 'DFP', 'BFGS']

for method in methods:
    x_min, f_min, iters = quasi_newton(f, grad_f, x0, tol=tol, alpha=alpha, beta=beta, method=method)
    print(f"Metoda: {method}")
    print(f"Minimum znalezione w punkcie: {x_min}")
    print(f"Wartość funkcji w minimum: {f_min}")
    print(f"Liczba iteracji: {iters}\n")
