import numpy as np
import forward_substitution
import backward_substitution
from gaussian_elimination_column import gaussian_elimination

A = np.zeros((84,84),dtype = float)

b = np.zeros(84, dtype= float)

b[0] = 7
b[83] = 14

for i in range(1,83):
    b[i] =15

for i in range(83):
    A[i,i] = 6
    A[i,i+1] = 1
    A[i+1,i] = 8

A[83,83] = 6

print(A)
print("\n")

print(b)
print("\n")
x, L, U, P = gaussian_elimination(A,b)
print("矩阵x为")
print(x)
