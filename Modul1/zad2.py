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

p = cp.Variable()
m = cp.Variable()
c = cp.Variable()

objective = cp.Minimize(0.15*p + 0.25*m + 0.05*c)

constraints = [
    (70*p + 121*m + 65*c) <= 2250,
    (70*p + 121*m + 65*c) >= 2000,
    (107*p + 500*m + 0*c) <= 10000,
    (107*p + 500*m + 0*c) >= 5000,
    (45*p + 40*m + 60*c) <= 1000,
    p <= 10,
    m <= 10,
    c <= 10,
    p >= 0,
    m >= 0,
    c >= 0
]

p1 = cp.Problem(objective, constraints)

p1.solve(verbose=True) #CLARABEL (Solver (including time spent in interface) took 1.000e-03 seconds) [p, m, c] = [6.5882353043431126, 9.999999998210054, 5.058823522233596]
#p1.solve(verbose=True, solver=cp.GLPK) # dual-simplex (Solver (including time spent in interface) took 2.996e-03 seconds) [p, m, c] = [6.58823529411765, 10.0, 5.05882352941176]
#p1.solve(verbose=True, solver=cp.ECOS) # interior-point (punktu wewnętrznego) (Solver (including time spent in interface) took 1.512e-03 seconds) [p, m, c] = [6.588235308925991, 9.999999999455817, 5.058823516359737]

print(" CVXPY Metoda 1 ")
print("p:", p.value)
print("m:", m.value)
print("c:", c.value)
print("--------------")

###########################

import numpy as np
import cvxpy as cp

x = cp.Variable(3)

c_obj = np.array([0.15, 0.25, 0.05])

A1 = np.array([[70, 121, 65],
               [70, 121, 65],
               [107, 500, 0],
               [107, 500, 0],
               [45, 40, 60]])

b1 = np.array([2250, 2000, 10000, 5000, 1000])
b2 = np.array([2250, 2000, 10000, 5000, 1000])

LB = np.array([0, 0, 0])
UB = np.array([10, 10, 10])

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
print("x1 =",p.value)
print("x2 =",m.value)
print("x3 =",c.value)
print("______________")





