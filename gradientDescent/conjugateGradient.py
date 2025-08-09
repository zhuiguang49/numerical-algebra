import numpy as np

def conjugate_gradient(A, b, x0=None, max_iter=1000, tol=1e-6):
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    r = b - A @ x  # 初始残差
    p = r.copy()   # 初始搜索方向
    residuals = [np.linalg.norm(r)]
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        
        # 收敛检查
        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)
        if residual_norm < tol:
            break
            
        # 计算新搜索方向
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        
    return {
        'x': x,
        'residuals': residuals,
        'iterations': k+1
    }