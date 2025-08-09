import numpy as np

def backward_substitution(A,b):
    n = len(b)
    x = np.zeros(n,dtype = float)

    for i in range(n-1,-1,-1):
        x[i]=(b[i]-np.dot(A[i,i+1:],x[i+1:]))/A[i,i]

    return x
