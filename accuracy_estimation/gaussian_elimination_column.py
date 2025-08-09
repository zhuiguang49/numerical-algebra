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

        if abs(A_copy[i,i]) < 1e-10:
            raise ValueError("选取列主元后，主对角线元素仍接近0，不满足高斯消元法可行性条件")

        A_copy[i+1:, i] = A_copy[i+1:, i] / A_copy[i, i]
        A_copy[i+1:, i+1:] = A_copy[i+1:, i+1:] - np.outer(A_copy[i+1:, i], A_copy[i, i+1:])


        print("这时的矩阵A为")
        print(A_copy)

    U = np.triu(A_copy)

    L = np.eye(n) + np.tril(A_copy, -1)

    y = forward_substitution.forward_substitution(L,np.matmul(P,b_copy))

    x = backward_substitution.backward_substitution(U,y)

    return x,L,U,P