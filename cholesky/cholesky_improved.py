import numpy as np
import forward_substitution
import backward_substitution

def cholesky_improved(A,b):
    n = len(A)
    L = np.eye(n)
    D = np.zeros(n)
    for i in range(n):
        D[i] = A[i,i] - np.dot(L[i,:i]**2,D[:i])
        dot_products = np.sum(L[i+1:n, :i] * L[i, :i] * D[:i], axis=1)
        L[i+1:n, i] = (A[i+1:n, i] - dot_products) / D[i]
    
    y = forward_substitution.forward_substitution(L,b)
    for i in range(n):
        y[i] = y[i]/D[i]
    x = backward_substitution.backward_substitution(L.T,y)

    return x,L,D
    
    


