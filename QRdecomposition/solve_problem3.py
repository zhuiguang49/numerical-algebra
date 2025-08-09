import numpy as np
from qrdecomposition import solve_linear_system

def build_system3():
    n = 40
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(n-1):
        for j in range(n-1):
            A[i, j] = 1 / (i + j + 1)
        b[i] = A[i, :].sum()
    
    A[-1, -1] = 1 / (2*(n-1) + 1)
    b[-1] = A[-1, :].sum()
    
    return A, b

if __name__ == "__main__":
    A, b = build_system3()
    
    # 使用正则化求解
    x = solve_linear_system(A, b, lambda_reg=1e-8)
    
    print("希尔伯特方程组的完整解:")
    print(x.round(6).tolist())
    print("\n残差范数:", np.linalg.norm(A @ x - b).round(6))