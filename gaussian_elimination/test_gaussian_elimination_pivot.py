import numpy as np
import forward_substitution
import backward_substitution
from gaussian_elimination_column import gaussian_elimination

def test_and_compare(A, b, test_name):
    """测试方程组并比较结果与numpy的解"""
    print(f"\n==== 测试案例: {test_name} ====")
    print(f"系数矩阵 A =\n{A}")
    print(f"右侧向量 b = {b}")
    
    try:
        # 使用您的高斯消元法
        x, L, U, P = gaussian_elimination(A, b)
        print(f"高斯消元解 x = {x}")
        
        # 验证 PA = LU
        print(f"验证 LU = PA: {np.allclose(np.dot(L, U), np.dot(P, A))}")
        
        # 验证解是否正确
        print(f"验证 Ax = b: {np.allclose(np.dot(A, x), b)}")
        
        # 与numpy解法比较
        x_numpy = np.linalg.solve(A, b)
        print(f"NumPy解 x = {x_numpy}")
        print(f"解的差异: {np.linalg.norm(x - x_numpy)}")
        
    except Exception as e:
        print(f"测试失败: {e}")

# 测试案例1: 标准3x3矩阵
A1 = np.array([
    [2.0, 1.0, 1.0],
    [1.0, 3.0, 2.0],
    [3.0, 2.0, 4.0]
], dtype=float)
b1 = np.array([4.0, 7.0, 10.0], dtype=float)
test_and_compare(A1, b1, "标准3x3矩阵")

# 测试案例2: 需要行交换的矩阵
A2 = np.array([
    [0.1, 2.0, 3.0],
    [3.0, 1.0, 2.0],
    [1.0, 4.0, 1.0]
], dtype=float)
b2 = np.array([5.0, 10.0, 8.0], dtype=float)
test_and_compare(A2, b2, "需要行交换的矩阵")

# 测试案例3: 病态条件矩阵 (condition number很大)
A3 = np.array([
    [1.0, 1.0, 1.0],
    [1.0, 1.0 + 1e-10, 1.0],
    [1.0, 1.0, 1.0 + 1e-10]
], dtype=float)
b3 = np.array([3.0, 3.0 + 1e-10, 3.0 + 1e-10], dtype=float)
test_and_compare(A3, b3, "病态条件矩阵")

# 测试案例4: 更大的系统 (5x5)
A4 = np.array([
    [2.0, 1.0, 0.0, 0.0, 1.0],
    [1.0, 3.0, 2.0, 0.0, 0.0],
    [0.0, 2.0, 4.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 3.0, 2.0],
    [1.0, 0.0, 0.0, 2.0, 4.0]
], dtype=float)
b4 = np.array([4.0, 6.0, 7.0, 6.0, 7.0], dtype=float)
test_and_compare(A4, b4, "5x5矩阵")

# 测试案例5: 希尔伯特矩阵 (著名的病态矩阵)
n = 4
A5 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A5[i, j] = 1.0 / (i + j + 1)
b5 = np.sum(A5, axis=1)  # 使解为全1向量
test_and_compare(A5, b5, "4x4希尔伯特矩阵")

# 测试案例6: 三对角矩阵
n = 5
A6 = np.zeros((n, n))
for i in range(n):
    A6[i, i] = 2.0
    if i > 0:
        A6[i, i-1] = -1.0
    if i < n-1:
        A6[i, i+1] = -1.0
b6 = np.ones(n)
test_and_compare(A6, b6, "5x5三对角矩阵")

# 测试案例7: 随机矩阵
np.random.seed(42)
A7 = np.random.rand(6, 6) * 10
b7 = np.random.rand(6) * 10
test_and_compare(A7, b7, "6x6随机矩阵")

# 测试案例8: 接近奇异的矩阵 (行几乎线性相关)
A8 = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0 + 1e-10],
    [3.0, 5.0, 8.0]
], dtype=float)
b8 = np.array([6.0, 12.0 + 2e-10, 16.0], dtype=float)
test_and_compare(A8, b8, "接近奇异的矩阵")