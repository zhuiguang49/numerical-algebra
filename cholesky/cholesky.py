import numpy as np
import forward_substitution
import backward_substitution

def cholesky(A,b):
    n = len(A)
    L = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            if j == i:
                L[i,i] = np.sqrt(A[i,i]-np.dot(L[i,:i],L[i,:i]))
            else:
                L[i,j] = (A[i,j] - np.dot(L[i,:j] , L[j,:j])) / L[j,j]

    y = forward_substitution.forward_substitution(L,b)
    x = backward_substitution.backward_substitution(L.T,y)

    return x,L



