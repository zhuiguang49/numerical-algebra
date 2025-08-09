import numpy as np

def jacobi(A, b, x0=None, max_iter=100, tol=1e-6):
    # 检验一下参数维度，不过对于本次作业来说不是必要的
    if A.shape[0] != A.shape[1]:
        raise ValueError("系数矩阵A必须是方阵")
    if A.shape[0] != b.shape[0]:
        raise ValueError("b的维度必须与A的行数一致")
    
    x = x0.copy() if x0 is not None else np.zeros_like(b)
    
    D = np.diag(np.diag(A))
    A2 = A - D
    D_inv = np.diag(1 / np.diag(D)) 
    
    for _ in range(max_iter):
        x_new = D_inv @ (b - A2 @ x)
        if np.linalg.norm(x_new - x, np.inf) < tol: 
            break
        x = x_new.copy()
    
    return x