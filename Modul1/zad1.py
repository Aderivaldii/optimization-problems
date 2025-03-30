import numpy as np
import cvxpy as cp

"""
This script solves a linear optimization problem using linear programming (LP),
aiming to minimize a cost function subject to a set of linear constraints.

The problem involves finding optimal values for decision variables to minimize a given cost,
with constraints specifying minimum or maximum requirements for certain system parameters.

The script demonstrates three different methods of modeling and solving the LP problem:
1. Method 1 – defining scalar variables and individual constraints.
2. Method 2 – using a vector variable and matrix representation of constraints.
3. Method 3 – reformulating constraints by combining matrices (using additional identity matrices and their negations).

Author: https://github.com/Pit1000
Date: 4.03.2025
"""

x1 = cp.Variable()
x2 = cp.Variable()
x3 = cp.Variable()

objective = cp.Minimize(300*x1 + 500*x2 + 800*x3)

constraints = [
0.8*x1 + 0.3*x2 + 0.1*x3 >= 0.3,
0.01*x1 + 0.4*x2 + 0.7*x3 >= 0.7,
0.15*x1 + 0.1*x2 + 0.2*x3 >= 0.1,
x1 >= 0,
x2 >= 0,
x3 >= 0
]

p1 = cp.Problem(objective, constraints)
p1.solve(verbose=True) #CLARABEL (Solver (including time spent in interface) took 1.000e-03 seconds) [x1, x2, x3] = [-4.649828235507262e-10, 0.8235294131587056, 0.5294117633655838]
#p1.solve(verbose=True, solver=cp.GLPK) # dual-simplex (Solver (including time spent in interface) took 3.999e-03 seconds) [x1, x2, x3] = [0.0, 0.823529411764706, 0.5294117647058824]
#p1.solve(verbose=True, solver=cp.ECOS) # interior-point (punktu wewnętrznego) (Solver (including time spent in interface) took 1.998e-03 seconds) [x1, x2, x3] = [1.6406683667794733e-09, 0.8235294131587056, 0.5294117633655838]

print(" CVXPY Metoda 1 ")
print("x1 =",x1.value)
print("x2 =",x2.value)
print("x3 =",x3.value)
print("--------------")

#####################################################
n = 3
x = cp.Variable(n)

c = np.array([300, 500, 800])

A = np.array([
    [0.8, 0.3, 0.1],
    [0.01, 0.4, 0.7],
    [0.15, 0.1, 0.2]
])

b = np.array([0.3, 0.7, 0.1])

LB = np.array([0, 0, 0])

objective = cp.Minimize(c.T @ x)

constraints = [A @ x >= b, x >= LB]

p1 = cp.Problem(objective, constraints)
p1.solve()#verbose=True)

print(" CVXPY Metoda 2 ")
print("x1 =", x.value[0])
print("x2 =", x.value[1])
print("x3 =", x.value[2])
print("---------------")

###########################################################
n = 3
x = cp.Variable(n)

A = np.array([
    [-0.8, -0.3, -0.1],
    [-0.01, -0.4, -0.7],
    [-0.15, -0.1, -0.2],
])

b = np.array([-0.3, -0.7, -0.1])
c = np.array([300, 500, 800])

Aeq = np.array([])
beq = np.array([])

LB = np.array([0, 0, 0])
UB = np.array([np.inf, np.inf, np.inf])

AA = np.vstack((A, np.eye(n), -np.eye(n)))
bb = np.hstack((b, UB, -LB))

objective = cp.Minimize(c.T @ x)

constraints = [ AA @ x <= bb ]
p3 = cp.Problem(objective, constraints)
p3.solve()

print("--------------")
print("CVXPY Method 3")
print("x1 =", x.value[0])
print("x2 =", x.value[1])
print("x3 =", x.value[2])
print("--------------")




