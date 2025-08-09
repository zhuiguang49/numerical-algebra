import numpy as np
from qrdecomposition import solve_linear_system

def build_system1():
    n = 84
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n)
    
    # 构造矩阵A
    for i in range(n-1):
        A[i, i] = 6
        A[i, i+1] = 1
        A[i+1, i] = 8
    A[-1, -1] = 6
    
    # 构造向量b
    b[0] = 7
    b[-1] = 14
    b[1:-1] = 15
    
    return A, b

if __name__ == "__main__":
    A, b = build_system1()
    x = solve_linear_system(A, b) 
    
    print("第一个方程组的完整解:")
    print(x.round(6))
    print("\n残差范数:", np.linalg.norm(A @ x - b).round(12))