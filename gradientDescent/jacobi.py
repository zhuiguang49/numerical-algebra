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
    
    # 添加残差记录
    residuals = []
    r = b - A @ x
    residuals.append(np.linalg.norm(r))
    
    iter_count = 0
    for iter_count in range(max_iter):
        x_new = D_inv @ (b - A2 @ x)
        
        # 计算残差并记录
        r = b - A @ x_new
        residuals.append(np.linalg.norm(r))
        
        if np.linalg.norm(x_new - x, np.inf) < tol: 
            break
        x = x_new.copy()
    
    # 返回结果字典，包含解向量、残差记录和迭代次数
    return {
        'x': x,
        'residuals': residuals,
        'iterations': iter_count + 1
    }