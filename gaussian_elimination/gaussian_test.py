import numpy as np
import forward_substitution
import backward_substitution

def gaussian_elimination(A,b):
    A_copy = np.copy(A)
    b_copy = np.copy(b)
    n = len(b_copy)
    
    # 这里我们是用一个置换矩阵来跟踪行的交换，所以要记录一下
    P = np.eye(n) 

    for i in range(n-1):
        max_row = i
        for j in range(i+1,n):
            if abs(A_copy[j,i]) > abs(A_copy[max_row,i]):
                max_row = j
        if max_row != i:
            A_copy[[i, max_row], :] = A_copy[[max_row, i], :]
            P[[i, max_row], :] = P[[max_row, i], :]
            b_copy[[i, max_row]] = b_copy[[max_row, i]]  
            
        

        A_copy[i+1:,i]=A_copy[i+1:,i]/A_copy[i,i]
        A_copy[i+1:,i+1:]=A_copy[i+1:,i+1:]-np.outer(A_copy[i+1:,i],A_copy[i,i+1:])
        b_copy[i+1:]=b_copy[i+1:]-b_copy[i]*A_copy[i+1:,i]


    U = np.triu(A_copy)
    L = np.tril(A_copy,-1)+np.eye(n)

    x = backward_substitution.backward_substitution(U,b_copy)

    return x,L,U,P