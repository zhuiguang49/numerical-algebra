import numpy as np

def gauss_seidel(A, b, x0=None, max_iter=100, tol=1e-6):
    # 检验一下参数维度，不过对于本次作业来说不是必要的
    if A.shape[0]!= A.shape[1]:
        raise ValueError("系数矩阵A必须是方阵")
    if A.shape[0]!= b.shape[0]:
        raise ValueError("b的维度必须与A的行数一致")

    x = x0.copy() if x0 is not None else np.zeros_like(b)
    n = A.shape[0]
    
    # 添加残差记录
    residuals = []
    r = b - A @ x
    residuals.append(np.linalg.norm(r))
    
    iter_count = 0
    for iter_count in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            lower_sum = A[i, :i] @ x[:i]
            upper_sum = A[i, i+1:] @ x_prev[i+1:]
            x[i] = (b[i] - lower_sum - upper_sum) / A[i, i]
        
        # 计算残差并记录
        r = b - A @ x
        residuals.append(np.linalg.norm(r))
        
        if np.linalg.norm(x - x_prev, np.inf) < tol:
            break
    
    # 返回结果字典，包含解向量、残差记录和迭代次数
    return {
        'x': x,
        'residuals': residuals,
        'iterations': iter_count + 1
    }