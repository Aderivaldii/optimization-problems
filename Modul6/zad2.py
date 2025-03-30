import numpy as np
import matplotlib.pyplot as plt

"""
The script performs iterative minimization of a function using the damped Newton's method,
where the step length is determined via backtracking line search.
The implementation compares the algorithm's behavior for different values
of the parameter t (0.1, 1.0, 10.0), allowing observation of how parameterization
of the objective function influences iteration progression and method convergence.

Author: https://github.com/Pit1000
"""

P = (1.0 / 8.0) * np.array([
    [7.0, np.sqrt(3)],
    [np.sqrt(3), 5.0]
])

x_c = np.array([1.0, 1.0])

# (19)
def f0(x, t):
    # Sprawdzenie dziedziny (log wymaga argumentu > 0)
    d = x - x_c
    domain_val = 1.0 - d.T @ P @ d
    if domain_val <= 0.0:
        return float('inf')

    val_exp = t * (np.exp(x[0] + 3.0*x[1] - 0.1) + np.exp(-x[0] - 0.1))
    val_log = -np.log(domain_val)
    return val_exp + val_log

# (20)
def grad_f0(x, t):
    d = x - x_c
    domain_val = 1.0 - d.T @ P @ d
    if domain_val <= 0.0:
        return np.array([0.0, 0.0])

    exp1 = np.exp(x[0] + 3.0*x[1] - 0.1)
    exp2 = np.exp(-x[0] - 0.1)

    grad_exp_x1 = t * (exp1 - exp2)
    grad_exp_x2 = t * (3.0 * exp1)

    grad_log = 2.0 * P @ d / domain_val

    return np.array([grad_exp_x1, grad_exp_x2]) + grad_log

# (21)
def hess_f0(x, t):
    d = x - x_c
    domain_val = 1.0 - d.T @ P @ d
    if domain_val <= 0.0:
        return np.zeros((2, 2))

    exp1 = np.exp(x[0] + 3.0*x[1] - 0.1)
    exp2 = np.exp(-x[0] - 0.1)

    # t * ...
    h11_exp = t * (exp1 + exp2)
    h12_exp = t * (3.0 * exp1)
    h21_exp = t * (3.0 * exp1)
    h22_exp = t * (9.0 * exp1)
    H_exp = np.array([
        [h11_exp, h12_exp],
        [h21_exp, h22_exp]
    ])
    H1 = (2.0 / domain_val) * P
    diff_outer = np.outer(d, d)
    H2 = (4.0 / (domain_val**2)) * (P @ diff_outer @ P)

    H_c = H1 + H2

    return H_exp + H_c

# Metoda Newtona z tłumieniem (wersja C)
def damped_newton_method(x0, t, tol=1e-4, alpha=0.5, beta=0.5, max_iter=100):
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f0(x, t)]
    iter_count = 0

    while True:
        iter_count += 1

        g = grad_f0(x, t)
        H = hess_f0(x, t)

        v = -np.linalg.inv(H) @ g
        decrement = -g.T @ v

        if decrement < tol or iter_count >= max_iter:
            break

        s = 1.0
        while f0(x + s * v, t) > f0(x, t) + s * alpha * (g.T @ v):
            s *= beta

        x = x + s * v
        x_history.append(x.copy())
        f_history.append(f0(x, t))

    return x, np.array(x_history), np.array(f_history)

# Uruchomienie dla różnych wartości t = 0.1, 1.0, 10.0
x0 = x_c
t_values = [0.1, 1.0, 10.0]

for t_test in t_values:
    x_opt, x_hist, f_hist = damped_newton_method(x0, t_test, tol=1e-4, alpha=0.3, beta=0.8, max_iter=100)
    print(f"\n===== Wyniki dla t = {t_test} =====")
    print("  Rozwiązanie:", x_opt)
    print("  Ostateczna wartość f0:", f0(x_opt, t_test))
    print("  Liczba iteracji:", len(x_hist) - 1)

    plt.figure()
    iterations = np.arange(0, len(x_hist))
    plt.scatter(iterations, f_hist, label="Iteracje", marker="o", color="blue")
    plt.xlim(-0.2, len(x_hist)+1)
    plt.ylim(0, max(f_hist) + 10)
    plt.grid()
    plt.xlabel("$k$")
    plt.ylabel("$f_0(x_0, t)$")
    plt.title("$f_0(x_0, t)$ dla $t = $" + str(t_test))
    plt.legend()


    # Wykresy
    x_vals = np.linspace(-3.0, 3.0, 300)
    y_vals = np.linspace(-3.0, 3.0, 300)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Obliczamy wartości f0
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f0([X[i, j], Y[i, j]], t_test)

    levels = [1.5, 2.5, 4, 6, 8, 20, 50, 200, 600]

    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=levels, colors='black')
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_title("$f_0(x_0, t)$ dla $t = $" + str(t_test))
    ax.set_aspect("equal", adjustable="box")
    ax.plot(x_hist[:, 0], x_hist[:, 1], 'o-', color='blue')
    plt.show()





