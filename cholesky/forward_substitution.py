import numpy as np

def forward_substitution(A,b):
    n = len(b)
    x = np.zeros(n,dtype = float)

    for i in range(n):
        x[i] = (b[i]-np.dot(A[i,:i],x[:i]))/A[i,i]

    return x



