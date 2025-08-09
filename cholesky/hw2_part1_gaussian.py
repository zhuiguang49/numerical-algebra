import numpy as np
import forward_substitution
import backward_substitution
import gaussian_test

A = np.zeros((100,100))
for i in range(99):
    A[i,i] = 10
    A[i,i+1] = 1
    A[i+1,i] = 1

A[99,99] = 10

b = np.random.rand(100)

x,L,U,P = gaussian_test.gaussian_elimination(A,b)

print("此时的b为:")
print(b)

print("第一小问的解为：")

print(x)