import numpy as np
import matplotlib.pyplot as plt

'''
This code implements the backtracking line search method—a technique for selecting step length in iterative optimization algorithms.

It uses the Armijo condition guaranteeing that the selected step leads to a reduction in the function value:
    φ(s) ≤ φ(0) + α * s * φ'(0), 
    where:
        φ(s) is the objective function at step length s.
        α ∈ (0, 1), controlling the sufficient decrease condition.
        φ'(0) is the derivative of the objective function at s = 0.
        
The algorithm starts with an initial value s_init and iteratively reduces s by multiplying it by β ∈ (0, 1), until the condition is satisfied.

For demonstration, the method is applied to two functions:
    - phi1 (quadratic function),
    - phi2 (cubic function),
along with their derivatives.

Author: https://github.com/Pit1000
'''

def phi1(s):
    return 20 * s**2 - 44 * s + 29

def dphi1(s):
    return 40 * s - 44

def phi2(s):
    return 40 * s**3 + 20 * s**2 - 44 * s + 29

def dphi2(s):
    return 120 * s**2 + 40 * s - 44

def backtracking_line_search(phi, dphi, alpha, beta, s_init):

    phi_0 = phi(0.0)
    dphi_0 = dphi(0.0)

    s = s_init
    s_iterations = []

    while True:
        s_iterations.append(s)
        # Warunek Armijo
        if phi(s) <= phi_0 + alpha * s * dphi_0:
            break
        else:
            s *= beta

    return s, s_iterations

# Wywołanie funckcji dla phi1 i phi2
alpha_1 = 0.3
alpha_2 = 0.4
beta = 0.9
s_init = 1.0

s_final_1, s_iters_1 = backtracking_line_search(phi1, dphi1, alpha=alpha_1, beta=beta, s_init=s_init)
print(f"[phi1(s)] Ostateczny krok s = {s_final_1:.5f}")
print(f"Iteracje: {s_iters_1}")

s_final_2, s_iters_2 = backtracking_line_search(phi2, dphi2, alpha=alpha_2, beta=beta, s_init=s_init)
print(f"[phi2(s)] Ostateczny krok s = {s_final_2:.5f}")
print(f"Iteracje: {s_iters_2}")

# Wykres dla Phi1
s_values = np.linspace(0, 2.5, 200)
phi_0_1 = phi1(0)
dphi_0_1 = dphi1(0)

y_bar_s_1 = phi_0_1 + alpha_1 * dphi_0_1 * s_values

plt.figure()
plt.plot(s_values,
         [phi1(s) for s in s_values],
         label=r'$\phi_1(s) = 20s^2 - 44s + 29$')

plt.plot(s_values, y_bar_s_1, label=fr'$y(s), \alpha={alpha_1}$', linestyle='solid')
plt.scatter(s_iters_1, [phi1(si) for si in s_iters_1], marker='o', label='Iteracje')
plt.axvline(s_final_1, linestyle='--',
            label=f"s final = {s_final_1:.3f}")

alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for a in alpha_values:
    y_bar_s_temp = phi_0_1 + a * dphi_0_1 * s_values
    plt.plot(s_values, y_bar_s_temp, linestyle='dotted',
             label=fr'$y(s), \alpha={a}$')

plt.ylim([-20, 50])
plt.xlim([0, 2.5])
plt.title(r"Backtracking line search $\phi_1(s)$")
plt.xlabel("s")
plt.ylabel("$\\phi(s)$ i $y(s)$")
plt.grid(True)
plt.legend()
plt.show()

# Wykres dla Phi2
phi_0_2 = phi2(0)
dphi_0_2 = dphi2(0)

y_bar_s_2 = phi_0_2 + alpha_2 * dphi_0_2 * s_values

plt.figure()
plt.plot(s_values,
         [phi2(s) for s in s_values],
         label=r'$\phi_2(s) = 40s^3 + 20s^2 - 44s + 29$', linestyle='solid')
plt.plot(s_values, y_bar_s_2,
         label=fr'$y(s), \alpha={alpha_2}$', linestyle='solid')
plt.scatter(s_iters_2, [phi2(si) for si in s_iters_2], marker='o', label='Iteracje')
plt.axvline(s_final_2, linestyle='--', label=f"s final = {s_final_2:.3f}")

alpha_values_2 = [0.0, 0.2, 0.6, 0.8, 1.0]
for a in alpha_values_2:
    y_bar_s_temp_2 = phi_0_2 + a * dphi_0_2 * s_values
    plt.plot(s_values, y_bar_s_temp_2, linestyle='dotted',
             label=fr'$y(s), \alpha={a}$')

plt.ylim([-20, 50])
plt.xlim([0, 2.5])
plt.title(r"Backtracking line search $\phi_2(s)$")
plt.xlabel("s")
plt.ylabel("$\\phi(s)$ i $y(s)$")
plt.grid(True)
plt.legend()
plt.show()
