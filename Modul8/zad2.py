import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

'''
Nonlinear constrained least squares problem solved using 
the Augmented Lagrangian Algorithm (ALA) with Levenberg–Marquardt (LM) method.
Performs ALA iterations combined with a penalty method,
calculating the Feasibility Residual (FR) and Optimality condition Residual (OR)
as functions of cumulative LM iterations.

Author: https://github.com/Pit1000
'''

# (42a)
def f_func(x):
    return x - np.array([1.0, 1.0, 1.0])
# (42b)
def g_func(x):
    g1 = x[0]**2 + 0.5*(x[1]**2) + x[2]**2 - 1.0
    g2 = (0.8*x[0]**2 + 2.5*x[1]**2 + x[2]**2
          + 2*x[0]*x[2] - x[0] - x[1] - x[2] - 1.0)
    return np.array([g1, g2])

def jac_f(x):
    return np.eye(3)

def jac_g(x):
    g1_x1 = 2.0*x[0]
    g1_x2 = 1.0*x[1]
    g1_x3 = 2.0*x[2]

    g2_x1 = 1.6*x[0] + 2.0*x[2] - 1.0
    g2_x2 = 5.0*x[1] - 1.0
    g2_x3 = 2.0*x[0] + 2.0*x[2] - 1.0

    return np.array([
        [g1_x1, g1_x2, g1_x3],
        [g2_x1, g2_x2, g2_x3]
    ])

def FR(x):
    """
    residuum warunku dopuszczalnosci
    Feasibility residual = ||g(x)||_2
    """
    return norm(g_func(x), 2)

def OR_augm(x, z):
    """
    residuum warunku optymalnosci
    OR (optimality residual) dla Augmented Lagrangian:
    """
    term1 = 2.0 * f_func(x)  # (3,)
    Jg = jac_g(x)            # (2,3)
    term2 = Jg.T @ z         # (3,)
    return norm(term1 + term2, 2)

def OR_penalty(x, mu):
    """
    OR (optimality residual) w metodzie kary:
    """
    term1 = 2.0 * f_func(x)       # (3,)
    term2 = 2.0*mu * (jac_g(x).T @ g_func(x))
    return norm(term1 + term2, 2)

def residual_ALA(x, mu, z):
    r_f = f_func(x)  # (3,)
    r_g = g_func(x) + z/(2.0*mu)  # (2,)
    return np.concatenate([r_f, np.sqrt(mu)*r_g])

def residual_penalty(x, mu):
    r_f = f_func(x)        # (3,)
    r_g = np.sqrt(mu)*g_func(x)  # (2,)
    return np.concatenate([r_f, r_g])

def ALA(x0 = np.zeros(3), z0 = np.zeros(2), mu0 = 1.0, tol = 1e-5, max_outer = 50):
    x = x0.copy()
    z = z0.copy()
    mu = mu0

    fr_list = []
    or_list = []
    mu_list = []
    lm_iters = []
    cum_lm = 0

    while True:
        fr_val = FR(x)
        or_val = OR_augm(x, z)
        fr_list.append(fr_val)
        or_list.append(or_val)
        mu_list.append(mu)
        lm_iters.append(cum_lm)

        if fr_val < tol and or_val < tol:
            break
        if len(fr_list) > max_outer:
            break

        sol = least_squares(residual_ALA, x, args=(mu, z), method='lm')
        x_new = sol.x

        n_lm = sol.njev if sol.njev is not None else sol.nfev
        if n_lm is None:
            n_lm = 1
        cum_lm += n_lm

        z_new = z + 2.0*mu*( g_func(x_new) + z/(2.0*mu) )

        g_norm_old = norm(g_func(x),2)
        g_norm_new = norm(g_func(x_new),2)
        if g_norm_new < 0.25*g_norm_old:
            mu_new = mu
        else:
            mu_new = 2.0*mu

        x = x_new
        z = z_new
        mu = mu_new

    return x, z, mu, np.array(fr_list), np.array(or_list), np.array(mu_list), np.array(lm_iters)

def penalty(
    x0 = np.zeros(3),
    mu0 = 1.0,
    tol = 1e-5,
    max_outer = 50
):
    x = x0.copy()
    mu = mu0

    fr_list = []
    or_list = []
    mu_list = []
    cum_lm_iters = []
    cum_lm = 0

    while True:
        fr_val = FR(x)
        or_val = OR_penalty(x, mu)
        fr_list.append(fr_val)
        or_list.append(or_val)
        mu_list.append(mu)
        cum_lm_iters.append(cum_lm)

        if fr_val < tol and or_val < tol:
            break
        if len(fr_list) > max_outer:
            break

        sol = least_squares(residual_penalty, x, args=(mu,), method='lm')
        x_new = sol.x

        n_lm = sol.njev if sol.njev is not None else sol.nfev
        if n_lm is None:
            n_lm = 1
        cum_lm += n_lm

        g_norm_old = norm(g_func(x),2)
        g_norm_new = norm(g_func(x_new),2)
        if g_norm_new < 0.25*g_norm_old:
            mu_new = mu
        else:
            mu_new = 2.0*mu

        x = x_new
        mu = mu_new

    return x, mu, np.array(fr_list), np.array(or_list), np.array(mu_list), np.array(cum_lm_iters)

x0 = np.zeros(3)
z0 = np.zeros(2)
mu0 = 1.0

x_ALA, z_ALA, mu_ALA, fr_ALA, or_ALA, mu_hist_ALA, lm_ALA = ALA(
    x0, z0, mu0, tol=1e-5, max_outer=50
)

final_obj_ALA = norm(f_func(x_ALA))**2
final_g_ALA = norm(g_func(x_ALA))

print("    AUGMENTED LAGRANGIAN (ALA)    ")
print(f"Rozwiazanie x* = {x_ALA}")
print(f"||f(x*)||^2   = {final_obj_ALA:.6e}")
print(f"||g(x*)||     = {final_g_ALA:.6e}")
print(f"z*            = {z_ALA}")
print(f"mu*           = {mu_ALA}")
print()

x_PEN, mu_PEN, fr_PEN, or_PEN, mu_hist_PEN, lm_PEN = penalty(
    x0, mu0, tol=1e-5, max_outer=50
)

final_obj_PEN = norm(f_func(x_PEN))**2
final_g_PEN = norm(g_func(x_PEN))

print("    PENALTY METHOD    ")
print(f"Rozwiazanie x* = {x_PEN}")
print(f"||f(x*)||^2   = {final_obj_PEN:.6e}")
print(f"||g(x*)||     = {final_g_PEN:.6e}")
print(f"mu*           = {mu_PEN}")
print()

# Wykresy
fig, axs = plt.subplots(1, 2, figsize=(12,5), sharey=False)

# ALA
axs[0].plot(lm_ALA, fr_ALA, 'o-b', label='FR')
axs[0].plot(lm_ALA, or_ALA, 's-r', label='OR')
axs[0].set_xlabel("cumulative LM iterations")
axs[0].set_ylabel("residual (FR, OR)")
axs[0].set_yscale('log')
axs[0].grid(True, which='both')
axs[0].legend()
axs[0].set_title("ALA - Augmented Lagrangian")

ax_mu = axs[0].twinx()
ax_mu.plot(lm_ALA, mu_hist_ALA, 'x--g', label='mu')
ax_mu.set_ylabel("mu", color='g')
ax_mu.set_yscale('log')
ax_mu.tick_params(axis='y', labelcolor='g')

# Penalty
axs[1].plot(lm_PEN, fr_PEN, 'o-b', label='FR')
axs[1].plot(lm_PEN, or_PEN, 's-r', label='OR')
axs[1].set_xlabel("cumulative LM iterations")
axs[1].set_ylabel("residual (FR, OR)")
axs[1].set_yscale('log')
axs[1].grid(True, which='both')
axs[1].legend()
axs[1].set_title("Penalty")

ax_mu2 = axs[1].twinx()
ax_mu2.plot(lm_PEN, mu_hist_PEN, 'x--g', label='mu')
ax_mu2.set_ylabel("mu", color='g')
ax_mu2.set_yscale('log')
ax_mu2.tick_params(axis='y', labelcolor='g')

plt.tight_layout()
plt.show()
