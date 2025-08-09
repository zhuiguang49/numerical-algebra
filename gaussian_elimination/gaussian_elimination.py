import numpy as np
import forward_substitution
import backward_substitution

def gaussian_elimination(A,b):
    A_copy = np.copy(A)
    b_copy = np.copy(b)
    n = len(b_copy)
    
    for i in range(n-1):
        if abs(A_copy[i,i]) < 1e-10:
            raise ValueError("主对角线元素接近0，不满足高斯消元法可行性条件")
        A_copy[i+1:,i]=A_copy[i+1:,i]/A_copy[i,i]
        A_copy[i+1:,i+1:]=A_copy[i+1:,i+1:]-np.outer(A_copy[i+1:,i],A_copy[i,i+1:])

    U = np.triu(A_copy)
    L = np.tril(A_copy,-1)+np.eye(n)

    y = forward_substitution.forward_substitution(L, b_copy)
    x = backward_substitution.backward_substitution(U, y)

    return x, L, U, np.eye(n) # 这里最后一个返回值我们返回的是置换矩阵（和选取列主元的情况对应起来），因为这是普通的高斯消元法，并不需要进行行交换


