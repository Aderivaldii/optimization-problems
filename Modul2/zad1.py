import cvxpy
import matplotlib.pyplot as plt
import scipy.io
from dataclasses import dataclass
from typing import Any, List

"""
This code implements a solution to an isoperimetric optimization problem. 
The objective is to determine a function f(x) that maximizes the area under the curve (the integral of f(x)), subject to the following constraints:
  - Total curve length (sum of discrete segment lengths must not exceed L)
  - Maximum curvature (discrete approximation of the second derivative bounded by constant C)
  - Enforcement of passing through predefined points (f(0)=0, f(a)=0, and additional points defined by F and y_fixed)

Problem parameters (a, C, L, N, F, y_fixed) are read from the file "isoPerimData.mat".

Implementation details:
  - CVXPY is used for modeling and solving the convex optimization problem.
  - SciPy is utilized for loading data from the file.
  - Matplotlib is employed for visualization of the resulting curve along with fixed points.
  - The Point dataclass clearly stores the coordinates of points.

The interval [0, a] is discretized into N+1 points. The objective function (maximizing the sum of function values) is formulated, constraints are added accordingly, and finally, the optimization problem is solved.

Author: https://github.com/Pit1000
"""

@dataclass
class Point:
    x: float
    y: Any

data = scipy.io.loadmat("isoPerimData.mat")
a = data["a"][0][0]
C = data["C"][0][0]
L = data["L"][0][0]
N = data["N"][0][0]
F = [i[0] for i in data["F"]]
y_fixed = [i[0] for i in data["y_fixed"]]

h = a / N
f_points: List[Point] = []

for i in range(N + 1):
    x_i = i * h

    if i in (0, N):
        y_i = 0
    elif i + 1 in F:
        y_i = y_fixed[i]
    else:
        y_i = cvxpy.Variable()#nonneg=True) #(b)

    f_points.append(Point(x_i, y_i))

discrete_lengths: List[Any] = []

for i in range(N):
    vec = cvxpy.vstack([h, f_points[i + 1].y - f_points[i].y])
    length = cvxpy.norm(vec, 2)
    discrete_lengths.append(length)

length_constraint = cvxpy.sum(discrete_lengths) <= L

curvature_constraints: List[Any] = []

for i in range(N - 1):
    curvature_constraint = cvxpy.abs((f_points[i + 2].y - 2 * f_points[i + 1].y + f_points[i].y) / h**2) <= C
    curvature_constraints.append(curvature_constraint)

objective = cvxpy.Maximize(h * cvxpy.sum([i.y for i in f_points]))
#(a)
#objective = cvxpy.Minimize(h * cvxpy.sum([i.y for i in f_points]))
problem = cvxpy.Problem(objective, [length_constraint] + curvature_constraints)
problem.solve()

#print("status:", problem.status)
print(f"A = {round(problem.value, 4)}")
#print(f"A = {problem.value}" )

y_computed = [i.y.value if type(i.y) is cvxpy.Variable else i.y for i in f_points]

plt.plot([i.x for i in f_points], y_computed, marker=",", color="black", linestyle="solid")
plt.plot([f_points[f - 1].x for f in F], [y_fixed[f - 1] for f in F], marker="o", color="black", linestyle="None")

plt.xlabel("x / a")
plt.ylabel("y(x)")

plt.grid(True)

plt.show()