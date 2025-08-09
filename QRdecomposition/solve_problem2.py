import numpy as np
from qrdecomposition import solve_linear_system

def build_system2():
    n = 100
    A = np.zeros((n, n))
    b = np.random.rand(n)
    
    # 构造矩阵A
    for i in range(n-1):
        A[i, i] = 10
        A[i, i+1] = 1
        A[i+1, i] = 1
    A[-1, -1] = 10
    
    return A, b

if __name__ == "__main__":
    A, b = build_system2()
    x = solve_linear_system(A, b)
    
    print("第二个方程组的完整解:")
    print(x.round(6))
    print("\n均值:", x.mean().round(4))
    print("标准差:", x.std().round(4))
    print("最大残差:", np.abs(A @ x - b).max().round(12))