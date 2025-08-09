import numpy as np

def sor(A, b, omega, x0=None, max_iter=100, tol=1e-6):
    # 检验一下参数维度，不过对于本次作业来说不是必要的
    if A.shape[0] != A.shape[1]:
        raise ValueError("系数矩阵A必须是方阵")
    if A.shape[0] != b.shape[0]:
        raise ValueError("b的维度必须与A的行数一致")
    
    x = x0.copy() if x0 is not None else np.zeros_like(b)
    n = A.shape[0]
    
    for _ in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            gs_update = (b[i] - A[i,:i] @ x[:i] - A[i,i+1:] @ x_prev[i+1:]) / A[i,i]
            x[i] = (1 - omega) * x_prev[i] + omega * gs_update
        
        if np.linalg.norm(x - x_prev, np.inf) < tol:
            break
    
    return x