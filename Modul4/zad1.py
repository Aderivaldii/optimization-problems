import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

"""
This code implements sparse signal representation using the basis pursuit method.

In the first part, a signal with varying amplitude and phase is defined, followed by constructing a Gabor dictionary of basis functions,
 consisting of Gaussian pulses modulated by cosine and sine functions for different values of time (τ) and frequency (ω).

Main processing steps:
- Constructing matrix X composed of Gabor basis functions (an overcomplete dictionary).
- Applying L1 regularization (Lasso model) to select significant basis functions, yielding a sparse coefficient vector.
- Performing further fitting using least squares for the selected bases to approximate the original signal.
- Evaluating the approximation quality by calculating mean squared error (MSE) and relative error.
- Visualizing the original signal, its approximation, and performing a time-frequency analysis (displaying selected dictionary elements against instantaneous frequency).

Author: https://github.com/Pit1000
"""

n_samples = 501
t = np.linspace(0, 1, n_samples)

# Definicja Sygnału
def original_signal(t):
    a = 1.0 + 0.5*np.sin(11.0*t)
    theta = 30.0*np.sin(5.0*t)
    return a * np.sin(theta)

y = original_signal(t)

tau_values = np.arange(0.0, 1.0, 0.002)
omega_values = np.arange(0, 155, 5)
sigma = 0.05

X = np.zeros((n_samples, len(tau_values)*len(omega_values)*2), dtype=float)

#Uzupełnianie macierzy 501 × 61 = 30561
#Budowanie słownika funkcji bazowych Gabora
col_index = 0
for w in omega_values:
    if w == 0:
        # tylko faza kosinusa (tu w=0 => cos(0)=1)
        for tau in tau_values:
            gauss = np.exp(-((t - tau)**2)/(sigma**2))
            X[:, col_index] = gauss
            col_index += 1
    else:
        # dwie fazy: cos i sin
        for tau in tau_values:
            gauss = np.exp(-((t - tau)**2)/(sigma**2))
            X[:, col_index] = gauss * np.cos(w*t)
            col_index += 1
            X[:, col_index] = gauss * np.sin(w*t)
            col_index += 1

#print("Shape Matrix X = ", X.shape)

# (1) Regularyzacja - które kolumny X - funcje bazowe sa istotne
alpha_val = 0.01
lasso_model = Lasso(alpha=alpha_val, fit_intercept=False, max_iter=2000, tol=1e-4)
lasso_model.fit(X, y)
x_l1 = lasso_model.coef_

nonzero_indices = np.flatnonzero(np.abs(x_l1) > 1e-12)
print("Liczba niezerowych współczynników:", len(nonzero_indices))

X_reduced = X[:, nonzero_indices]

#Najmniejsze kwadraty
w_refined, residuals, rank, svals = np.linalg.lstsq(X_reduced, y, rcond=None)
y_hat = X_reduced @ w_refined

#Ocena błędu i wykresy
mse = np.mean((y - y_hat)**2)
rel_error = np.sqrt(mse) / np.sqrt(np.mean(y**2))

print(f"Błąd średniokwadratowy MSE = {mse:.4e}")
print(f"Błąd względny (RMSE / RMS(y)) = {rel_error:.4e}")

# Wykres oryginalnego sygnału i aproksymacji
plt.figure()
plt.plot(t, y, label="Oryginalny sygnał")
plt.plot(t, y_hat, '--', label="Aproksymacja")
plt.legend()
plt.title("Porównanie sygnału oryginalnego i aproksymowanego")
plt.xlabel("t")
plt.ylabel("y")
plt.show()

# Wykres różnicy
plt.figure()
plt.plot(t, y - y_hat)
plt.title("Błąd aproksymacji y(t) - ŷ(t)")
plt.xlabel("t")
plt.ylabel("Δ")
plt.show()

#Analiza Czasowo-Częstotliwościowa
nonzeros_info = []
col_start = 0
#print(omega_values)
for w in omega_values:
    if w == 0:
        for i_tau, tau in enumerate(tau_values):
            c_idx = col_start + i_tau
            if c_idx in nonzero_indices:
                nonzeros_info.append((tau, w, 'c'))
        col_start += 501
    else:
        for i_tau, tau in enumerate(tau_values):
            cos_idx = col_start + 2*i_tau
            sin_idx = col_start + 2*i_tau + 1
            if cos_idx in nonzero_indices:
                nonzeros_info.append((tau, w, 'c'))
            if sin_idx in nonzero_indices:
                nonzeros_info.append((tau, w, 's'))
        col_start += 2*len(tau_values)

# nonzeros_info - lista krotek (tau, omega, faza)
tau_list = [item[0] for item in nonzeros_info]
omega_list = [item[1] for item in nonzeros_info]

# Wykres Analiza Czasowo-Częstotliwościowa
plt.figure()
plt.plot(t, 150*np.abs(np.cos(5*t)), '--', label='|dθ/dt| = 150|cos(5t)|')
plt.scatter(tau_list, omega_list, marker='o', s=20, facecolors='none', edgecolors='r', label='Wybrane bazy')
plt.legend()
plt.title("Analiza czasowo-częstotliwościowa")
plt.xlabel("τ")
plt.ylabel("ω")
plt.ylim([0, 160])
plt.show()
