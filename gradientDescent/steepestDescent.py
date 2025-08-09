import numpy as np

def steepest_descent(f, grad_f, x0, alpha=0.1, max_iter=1000, tol=1e-6, method='fixed'):
    x = x0.copy()
    history = {
        'points': [x0],
        'grad_norms': []
    }
    
    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        history['grad_norms'].append(grad_norm)
        
        # 停止条件检查
        if grad_norm < tol:
            break
            
        # 确定搜索方向
        direction = -grad
        
        # 步长选择
        if method == 'exact':
            # 精确线搜索
            phi = lambda a: f(x + a*direction)
            res = minimize_scalar(phi)
            alpha = res.x
        elif method == 'fixed':
            pass  # 直接使用给定alpha
        else:
            raise ValueError("无效的method参数，请选择'fixed'或'exact'")
            
        # 更新参数
        x = x + alpha * direction
        history['points'].append(x.copy())
        
    return {
        'x': x,
        'history': history,
        'grad_norm': grad_norm,
        'iterations': k+1
    }
