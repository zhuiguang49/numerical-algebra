import numpy as np
import forward_substitution
import backward_substitution
import cholesky
import cholesky_improved

A = np.zeros((40,40),dtype = float)
b = np.zeros(40)

for i in range(39):
    for j in range(39):
        A[i,j] = 1/(i+j+1)
    b[i] = np.sum(A[i,:])

A[39,39] = 1/79
b[39] = np.sum(A[39,:])

print("此时的b为:")
print(b)

x1,L1 = cholesky.cholesky(A,b)
print("第二小问平方根法的解为:")
print(x1)

x2,L2,D = cholesky_improved.cholesky_improved(A,b)
print("第二小问改进平方根法的解为:")
print(x2)