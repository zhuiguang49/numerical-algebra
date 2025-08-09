import numpy as np 
from backward_substitution import backward_substitution
from forward_substitution import forward_substitution
from gaussian_elimination_column import gaussian_elimination

A = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 10]],dtype=float)

b = np.array([1, 1, 1],dtype=float)

x, L, U, P = gaussian_elimination(A,b)


print("矩阵x为")
print(x)
print("\n")
print("矩阵L为")
print(L)
print("\n")
print("矩阵U为")
print(U)
print("\n")
print("矩阵P为")
print(P)