import numpy as np
import forward_substitution
import backward_substitution
import cholesky
import cholesky_improved

A = np.zeros((100,100))
for i in range(99):
    A[i,i] = 10
    A[i,i+1] = 1
    A[i+1,i] = 1

A[99,99] = 10

b = np.random.rand(100)

print("此时的b为:")
print(b)


x1,L1 = cholesky.cholesky(A,b)
print("第一小问平方根法的解为:")
print(x1)

x2,L2,D = cholesky_improved.cholesky_improved(A,b)
print("第一小问改进平方根法的解为:")
print(x2)










