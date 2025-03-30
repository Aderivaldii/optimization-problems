import numpy as np
import matplotlib.pyplot as plt

"""
The script performs iterative minimization of a function using two variants of Newton's method:
- The classic Newton's method, where the step can be interpreted as the Newton correction vector,
  i.e., the direction in which point x is updated, computed as the negative product of the inverse Hessian
  of the function and its gradient.
- The damped Newton's method, which selects step size s using a backtracking line search algorithm.

Author: https://github.com/Pit1000
"""

# (14)
def f0(x):
    x1, x2 = x[0], x[1]
    val = np.exp(x1 + 3.0 * x2 - 0.1) + np.exp(-x1 - 0.1) + (x - x_c).T @ P @ (x - x_c)
    return val

# (15)
P = (1.0 / 8.0) * np.array([[7.0, np.sqrt(3)],
                            [np.sqrt(3), 5.0]])
x_c = np.array([1.0, 1.0])

# (17) Gradient f0
def grad_f0(x):
    x1, x2 = x[0], x[1]
    df_dx1 = np.exp(x1 + 3.0 * x2 - 0.1) - np.exp(-x1 - 0.1)
    df_dx2 = 3.0 * np.exp(x1 + 3.0 * x2 - 0.1)
    df_vec = np.array([df_dx1, df_dx2]) + 2.0 * (P @ (x - x_c))
    return df_vec

# (18) Hesjan f0
def hess_f0(x):
    x1, x2 = x[0], x[1]
    exp1 = np.exp(x1 + 3.0 * x2 - 0.1)
    exp2 = np.exp(-x1 - 0.1)

    h11 = exp1 + exp2
    h12 = 3.0 * exp1
    h21 = 3.0 * exp1
    h22 = 9.0 * exp1

    H_exp = np.array([[h11, h12],
                      [h21, h22]])

    H_quad = 2.0 * P
    return H_exp + H_quad

# Metoda Newtona (wersja C)
def newton_method(x0, tol=1e-4, max_iter=100):

    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f0(x)]
    iter_count = 0

    while True:
        g = grad_f0(x)
        H = hess_f0(x)

        v = -np.linalg.inv(H) @ g

        decrement = -g.T @ v

        if decrement < tol or iter_count >= max_iter:
            break

        x = x + v
        x_history.append(x.copy())
        f_history.append(f0(x))

    return x, np.array(x_history), np.array(f_history)

# Metoda Newtona z tłumieniem (wersja C)
def damped_newton_method(x0, tol=1e-4, alpha=0.5, beta=0.5, max_iter=100):

    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f0(x)]
    iter_count = 0

    while True:
        iter_count += 1

        g = grad_f0(x)
        H = hess_f0(x)

        v = -np.linalg.inv(H) @ g

        decrement = -g.T @ v

        if decrement < tol or iter_count >= max_iter:
            break

        s = 1.0
        while f0(x + s * v) > f0(x) + s * alpha * (g.T @ v):
            s *= beta

        x = x + s * v
        x_history.append(x.copy())
        f_history.append(f0(x))

    return x, np.array(x_history), np.array(f_history)

# (16)
x0 = np.array([2.0, -2.0])

# Uruchomienie funkcji
x_opt_newton, x_hist_newton, f_hist_newton = newton_method(x0)

x_opt_damped, x_hist_damped, f_hist_damped = damped_newton_method(x0)

# Wyniki
print("Metoda Newtona:")
print("  Rozwiązanie:", x_opt_newton)
print("  Ostateczna wartość f0:", f0(x_opt_newton))
print("  Iteracje:", len(x_hist_newton) - 1)

print("\nMetoda Newtona z tłumieniem:")
print("  Rozwiązanie:", x_opt_damped)
print("  Ostateczna wartość f0:", f0(x_opt_damped))
print("  Iteracje:", len(x_hist_damped) - 1)

# Wykresy
x_vals = np.linspace(-3.0, 3.0, 100)
y_vals = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Obliczamy wartości f0
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f0([X[i, j], Y[i, j]])

levels = [1.5, 2.5, 4, 6, 8, 20, 50, 200, 600]

fig, ax = plt.subplots()
cs = ax.contour(X, Y, Z, levels=levels, colors='black')
ax.clabel(cs, inline=True, fontsize=8)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim([-3, 2])
ax.set_ylim([-2, 2])
ax.set_title("f0(x) - Klasyczny Newton")
ax.set_aspect("equal", adjustable="box")
ax.plot(x_hist_newton[:,0], x_hist_newton[:,1], 'o-', color='blue')

fig2, ax2 = plt.subplots()
cs = ax2.contour(X, Y, Z, levels=levels, colors='black')
ax2.clabel(cs, inline=True, fontsize=8)
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title("f0(x) - Newton z tłumieniem")
ax2.set_xlim([-3, 2])
ax2.set_ylim([-2, 2])
ax2.set_aspect("equal", adjustable="box")
ax2.plot(x_hist_damped[:,0], x_hist_damped[:,1], 'o-', color='red')
plt.show()

plt.figure()
plt.scatter(np.arange(len(f_hist_newton)), f_hist_newton, label="Klasyczny Newton", marker="o", color="blue")
plt.scatter(np.arange(len(f_hist_damped)), f_hist_damped,label="Newton z tłumieniem", marker="x", color="red")
plt.xlim(-0.2, 6.5)
plt.ylim(0, 20)
plt.grid()
plt.xlabel("$k$")
plt.ylabel("$f_0(x_k)$")
plt.legend()
plt.show()



