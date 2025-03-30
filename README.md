# Optimization Problems 

This project gathers experience from multiple optimization problems at the core of Machine Learning (ML), which, in turn, serve as the driving force behind broader concepts in Artificial Intelligence (AI). The repository is organized into modules, each addressing a distinct optimization topic and providing implementations in Python.

---
![ezgif com-animated-gif-maker(1)](https://github.com/user-attachments/assets/504761c2-3136-4242-879c-360c8005cd7c)

## Table of Contents
1. [Module 1 – Linear Programming (LP)](#module-1--linear-programming-lp)
2. [Module 2 – Isoperimetric Optimization](#module-2--isoperimetric-optimization)
3. [Module 3 – Piecewise Constant Signal Reconstruction](#module-3--piecewise-constant-signal-reconstruction)
4. [Module 4 – Basis Pursuit](#module-4--basis-pursuit)
5. [Module 5 – Backtracking Line Search](#module-5--backtracking-line-search)
6. [Module 6 – Newton's Method](#module-6--newtons-method)
7. [Module 7 – Levenberg–Marquardt Parameter Estimation](#module-7--levenbergmarquardt-parameter-estimation)
8. [Module 8 – Nonlinear Constrained Optimization](#module-8--nonlinear-constrained-optimization)
9. [Module 9 – Quasi-Newton Method](#module-9--quasi-newton-method)
10. [Module 10 – Linear Programming (LP) Solutions using Sequential Barrier Method (SBM)](#module-10--linear-programming-lp-solutions-using-sequential-barrier-method-sbm)
11. [Bibliography](#bibliography)

---

## Module 1 – Linear Programming (LP)

Examples of solving linear programming optimization problems using the **CVXPY** library in Python. The examples demonstrate different modeling techniques and solution approaches:
- Scalar vs. vector-based formulations
- Alternative transformations of constraints

---

## Module 2 – Isoperimetric Optimization

Implementation of an **isoperimetric optimization problem** using convex programming techniques. The objective is to determine a function \( f(x) \) that maximizes the area under the curve (the integral of \( f(x) \)) while satisfying:
- A specified total length of the curve
- A maximum curvature constraint
- Passing through given fixed points

---

## Module 3 – Piecewise Constant Signal Reconstruction

Two scripts that perform reconstruction of piecewise constant signals from noisy measurements:

1. **LASSO-based Optimization (`zad1.py`):**  
   - Uses the Least Absolute Shrinkage and Selection Operator (LASSO).  
   - Minimizes the \( L_2 \) norm of measurement errors with an \( L_1 \) norm constraint on the signal’s gradient to promote sparsity.

2. **Linear Programming-based Optimization (`zad2.py`):**  
   - Reformulates the signal reconstruction as a Linear Programming task.  
   - Minimizes discrepancies between the measured noisy signal and the estimated signal, enforcing piecewise constant behavior via linear constraints.

---

## Module 4 – Basis Pursuit

Implementation of a **Basis Pursuit** problem using an overcomplete dictionary of Gabor basis functions:
1. A synthetic signal is generated with varying amplitude and phase.
2. An overcomplete dictionary of Gabor functions is constructed.
3. **L1 regularization (Lasso)** selects a sparse subset of these functions that best represent the signal.
4. A refined approximation is obtained through a **least-squares fit** on the selected basis elements.
5. The code evaluates the reconstruction quality via metrics (e.g., mean squared error, relative error) and visualizes the time-frequency distribution of both the original and reconstructed signals.

---

## Module 5 – Backtracking Line Search

Implementation of the **Backtracking Line Search** algorithm, a common method for finding suitable step lengths in iterative optimization. It ensures a sufficient decrease in the objective function by checking the **Armijo condition**:

\[
f(x + s \, \Delta x) \leq f(x) + \alpha \, s \, \nabla f(x)^\top \Delta x
\]

Here:
- \( s \) is the step size to be determined.
- \( 0 < \alpha < 1 \) is the Armijo constant.
- \( \Delta x \) is the descent direction.

---

## Module 6 – Newton's Method

Comparison and implementation of two variations of Newton’s method for nonlinear optimization problems:

1. **Classic Newton’s Method**  
   - Uses the gradient and Hessian matrix to iteratively minimize a function.

2. **Damped Newton’s Method**  
   - Enhances the classic approach by incorporating a **backtracking line search** for better convergence properties.

---

## Module 7 – Levenberg–Marquardt Parameter Estimation

Implementation of the **Levenberg–Marquardt (LM)** algorithm to estimate parameters of various models:

1. **Sinusoidal Function (`zad1.py`)**  
   \[
   y(t) = A \,\sin(\omega t + \phi)
   \]  
   Estimated Parameters: Amplitude \( A \), Angular frequency \( \omega \), Phase shift \( \phi \)

2. **Damped Sinusoidal Function (`zad2.py`)**  
   \[
   y(t) = A \, e^{-a\, t} \sin(\omega t + \phi)
   \]  
   Estimated Parameters: Amplitude \( A \), Damping coefficient \( a \), Angular frequency \( \omega \), Phase shift \( \phi \)

3. **First-Order Inertia (`zad3.py`)**  
   \[
   y(t) = k \,\bigl( 1 - e^{-t/T} \bigr)
   \]  
   Estimated Parameters: Gain \( k \), Time constant \( T \)

4. **Double Inertia (`zad4.py`)**  
   \[
   y(t) = k \left[1 - \frac{1}{T_1 - T_2}\bigl(T_1 e^{-t/T_1} - T_2 e^{-t/T_2}\bigr)\right]
   \]  
   Estimated Parameters: Gain \( k \), Time constants \( T_1, T_2 \)

5. **Second-Order Oscillatory System (`zad5.py`)**  
   \[
   y(t) = k \Bigl[1 - e^{-\gamma t} \Bigl(\cos(\beta t) + \frac{\gamma}{\beta}\sin(\beta t)\Bigr)\Bigr]
   \]  
   Estimated Parameters: Gain \( k \), Damping factor \( \gamma \), Oscillation frequency \( \beta \)

---

## Module 8 – Nonlinear Constrained Optimization

Nonlinear constrained optimization algorithms using the **Augmented Lagrangian Algorithm (ALA)** combined with the **Levenberg–Marquardt (LM)** method:

- **`zad1.py`**  
  Solves a 2D nonlinear least squares problem with a single nonlinear constraint using ALA and LM. Includes residual visualization.

- **`zad2.py`**  
  Compares the Augmented Lagrangian Algorithm (ALA) and Penalty method for a 3D constrained optimization problem, visualizing residual convergence and parameter evolution.

- **`zad3.py`**  
  A **Boolean Least Squares** problem minimizing \( \|Ax - b\|^2 \) with elements of \( x \) restricted to +1 or -1. Compares brute-force and ALA solutions.

---

## Module 9 – Quasi-Newton Method

Scripts to find the minimum of a two-variable function using **quasi-Newton** optimization methods:

- **SR1 (Symmetric Rank One)**
- **DFP (Davidon–Fletcher–Powell)**
- **BFGS (Broyden–Fletcher–Goldfarb–Shanno)**

---

## Module 10 – Linear Programming (LP) Solutions using Sequential Barrier Method (SBM)

Python scripts demonstrating solutions to Linear Programming problems via the **Sequential Barrier Method (SBM)**. Results are compared with standard LP solvers (e.g., `linprog` from SciPy).

---

## Bibliography

1. A. Ben-Tal and A. Nemirovski.  
   **Lectures on Modern Convex Optimization.** SIAM, 2001.

2. Stephen Boyd and Lieven Vandenberghe.  
   **Convex Optimization.** Cambridge University Press, New York, NY, USA, 2004.  
   Available online:  
   [http://web.stanford.edu/~boyd/cvxbook/](http://web.stanford.edu/~boyd/cvxbook/)

3. Stephen Boyd and Lieven Vandenberghe.  
   **Additional Exercises for Convex Optimization,** 2004.  
   [https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook_extra_exercises.pdf](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook_extra_exercises.pdf)

4. G.C. Calafiore and L. El Ghaoui.  
   **Optimization Models.** Cambridge University Press, 2014.

5. E.K.P. Chong and S.H. Zak.  
   **An Introduction to Optimization.** Wiley, 2004.

6. Ulrich Münz, Amer Mešanović, Michael Metzger, and Philipp Wolfrum.  
   **Robust optimal dispatch, secondary, and primary reserve allocation for power systems with uncertain load and generation.**  
   IEEE Transactions on Control Systems Technology, 26(2):475–485, 2018.

7. Course materials from **Metody Optymalizacji** (Optimization Methods) for the Master’s program in Applied Computer Science at Warsaw University of Technology.

---

**Note:** Each module folder or script may include additional explanations, dependencies, and usage instructions. Please refer to the individual module documentation or script docstrings for more details.
