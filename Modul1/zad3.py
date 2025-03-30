import numpy as np
import cvxpy as cp

"""
This script solves a linear optimization problem using Linear Programming (LP), where the goal is to minimize a cost function while satisfying a set of linear constraints.

The problem involves finding optimal values for decision variables to minimize a given cost, where constraints specify minimum or maximum requirements for certain system parameters.

Two modeling approaches are presented in this code:
1. Scalar approach – each variable is defined separately, and constraints are formulated for individual expressions.
2. Vector approach – decision variables are combined into a single vector, and constraints are expressed in matrix form.

Author: https://github.com/Pit1000  
Date: 4.03.2025
"""

xlek1 = cp.Variable()
xlek2 = cp.Variable()
xsur1 = cp.Variable()
xsur2 = cp.Variable()

objective = cp.Minimize(100*xsur1 + 199.9*xsur2 + 700*xlek1 + 800*xlek2 - 6500*xlek1 - 7100*xlek2)

constraints = [
    0.01*xsur1 + 0.02*xsur2 - 0.5*xlek1 - 0.6*xlek2 >= 0,
    xsur1 + xsur2 <= 1000,
    90*xlek1 + 100*xlek2 <= 2000,
    40*xlek1 + 50*xlek2 <= 800,
    100*xsur1 + 199.9*xsur2 + 700*xlek1 + 800*xlek2 <= 100000,
    xsur1 >= 0,
    xsur2 >= 0,
    xlek1 >= 0,
    xlek2 >= 0
]

p1 = cp.Problem(objective, constraints)

p1.solve(verbose=True) #CLARABEL (Solver (including time spent in interface) took 1.000e-03 seconds) [xlek1, xlek2, xsur1, xsur2] = [17.551557695298804, 1.4434045670226588e-09, 4.409107600525186e-06, 438.7889402120869]
#p1.solve(verbose=True, solver=cp.GLPK) # dual-simplex (Solver (including time spent in interface) took 3.014e-03 seconds) [xlek1, xlek2, xsur1, xsur2] = [17.55155770074594, 2.960594732333751e-15, 0.0, 438.7889425186485]
#p1.solve(verbose=True, solver=cp.ECOS) # interior-point (punktu wewnętrznego) Solver (including time spent in interface) took 1.000e-03 seconds [xlek1, xlek2, xsur1, xsur2] = [17.551557696761897, 5.22042782065505e-10, 5.562279621739461e-06, 438.7889396532701]

print("CVXPY Metoda 1")
print("xlek1 = ", xlek1.value)
print("xlek2 = ", xlek2.value)
print("xsur1 = ", xsur1.value)
print("xsur2 = ", xsur2.value)
print("--------------")

###################################

x = cp.Variable(4)

c_obj = np.array([100, 199.9, -5800, -6300])

A1 = np.array([[0.01, 0.02, -0.5, -0.6],
               [1, 1, 0, 0],
               [0, 0, 90, 100],
               [0, 0, 40, 50],
               [100, 199.9, 700, 800]])

b1 = np.array([0, 1000, 2000, 800, 100000])
b2 = np.array([0, 1000, 2000, 800, 100000])

LB = np.array([0, 0, 0, 0])
UB = np.array([np.inf, np.inf, np.inf, np.inf])

objective = cp.Minimize(c_obj @ x)

constraints = [
    A1 @ x <= b1,
    A1 @ x >= b2,
    x >= LB,
    x <= UB
]

problem = cp.Problem(objective, constraints)
problem.solve()

print("CVXPY Metoda 2")
print("x1 =",xlek1.value)
print("x2 =",xlek2.value)
print("x3 =",xsur1.value)
print("x4 =",xsur2.value)
print("______________")

