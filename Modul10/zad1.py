import numpy as np
from scipy.optimize import linprog

def find_x(A, b, eps=1e-8):

    n = A.shape[1]
    c = np.zeros(n)

    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=[(None, None)] * len(c),
        method='highs',
        options={'maxiter': 1000}  # 'disp': True
    )

    return res.x

A = np.array([[1, 2],
            [3, 4],
            [-1, -1]])
b = np.array([10, 20, -3])

x = find_x(A, b)

print("Znaleziony wektor x:", x)
print("Ax =", A @ x)
print("b  =", b)

