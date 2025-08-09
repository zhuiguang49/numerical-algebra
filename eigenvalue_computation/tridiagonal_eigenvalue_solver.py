import numpy as np

def sturm_sequence(diag, sub_diag, x):
    """
    计算Sturm序列，用于统计小于x的特征值个数
    
    参数:
    diag: 主对角线元素
    sub_diag: 次对角线元素
    x: 测试值
    
    返回:
    小于x的特征值个数
    """
    n = len(diag)
    p = [1.0]  # p_0 = 1
    p.append(diag[0] - x)  # p_1 = a_1 - x
    
    # 计算Sturm序列
    for i in range(1, n):
        p_next = (diag[i] - x) * p[i] - sub_diag[i-1]**2 * p[i-1]
        p.append(p_next)
    
    # 统计符号变化次数
    sign_changes = 0
    for i in range(1, len(p)):
        if p[i-1] * p[i] < 0:
            sign_changes += 1
        elif p[i] == 0:
            # 处理零值情况
            if i < len(p) - 1:
                if p[i-1] * p[i+1] < 0:
                    sign_changes += 1
    
    return sign_changes

def bisection_eigenvalue(diag, sub_diag, k, tol=1e-12, max_iter=1000):
    """
    使用二分法求对称三对角矩阵的第k个特征值（从小到大排序）
    
    参数:
    diag: 主对角线元素数组
    sub_diag: 次对角线元素数组
    k: 求第k个特征值 (1-indexed)
    tol: 收敛容差
    max_iter: 最大迭代次数
    
    返回:
    第k个特征值
    """
    n = len(diag)
    
    # 估计特征值范围
    # Gershgorin圆盘定理估计边界
    min_bound = float('inf')
    max_bound = float('-inf')
    
    for i in range(n):
        center = diag[i]
        if i == 0:
            radius = abs(sub_diag[0]) if n > 1 else 0
        elif i == n-1:
            radius = abs(sub_diag[i-1])
        else:
            radius = abs(sub_diag[i-1]) + abs(sub_diag[i])
        
        min_bound = min(min_bound, center - radius)
        max_bound = max(max_bound, center + radius)
    
    # 扩大搜索范围
    range_width = max_bound - min_bound
    min_bound -= 0.1 * range_width
    max_bound += 0.1 * range_width
    
    # 二分法搜索第k个特征值
    left, right = min_bound, max_bound
    
    for iteration in range(max_iter):
        mid = (left + right) / 2
        count = sturm_sequence(diag, sub_diag, mid)
        
        if count < k:
            left = mid
        else:
            right = mid
        
        if abs(right - left) < tol:
            break
    
    return (left + right) / 2

def inverse_power_method(diag, sub_diag, eigenvalue, tol=1e-12, max_iter=1000):
    """
    使用反幂法求指定特征值对应的特征向量
    
    参数:
    diag: 主对角线元素数组
    sub_diag: 次对角线元素数组
    eigenvalue: 已知特征值
    tol: 收敛容差
    max_iter: 最大迭代次数
    
    返回:
    归一化的特征向量
    """
    n = len(diag)
    
    # 构造 (A - λI)
    # 由于是三对角矩阵，我们直接处理三对角线性方程组
    modified_diag = diag - eigenvalue
    
    # 初始向量
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for iteration in range(max_iter):
        v_old = v.copy()
        
        # 解 (A - λI)v_new = v_old
        # 使用Thomas算法求解三对角线性方程组
        v_new = thomas_algorithm(modified_diag, sub_diag, sub_diag, v_old)
        
        # 归一化
        v_new = v_new / np.linalg.norm(v_new)
        
        # 检查收敛性
        if np.linalg.norm(v_new - v_old) < tol:
            break
        
        v = v_new
    
    return v

def thomas_algorithm(a, b, c, d):
    """
    Thomas算法求解三对角线性方程组 Ax = d
    其中A的主对角线为a，下对角线为b，上对角线为c
    
    参数:
    a: 主对角线元素
    b: 下对角线元素 
    c: 上对角线元素
    d: 右端项
    
    返回:
    解向量x
    """
    n = len(a)
    
    # 前向消元
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    c_prime[0] = c[0] / a[0]
    d_prime[0] = d[0] / a[0]
    
    for i in range(1, n-1):
        c_prime[i] = c[i] / (a[i] - b[i-1] * c_prime[i-1])
        d_prime[i] = (d[i] - b[i-1] * d_prime[i-1]) / (a[i] - b[i-1] * c_prime[i-1])
    
    d_prime[n-1] = (d[n-1] - b[n-2] * d_prime[n-2]) / (a[n-1] - b[n-2] * c_prime[n-2])
    
    # 回代
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

def create_test_matrix(n):
    """
    创建测试矩阵A，主对角线为2，上下次对角线为-1
    
    参数:
    n: 矩阵阶数
    
    返回:
    diag: 主对角线
    sub_diag: 次对角线
    """
    diag = np.full(n, 2.0)
    sub_diag = np.full(n-1, -1.0)
    return diag, sub_diag

def verify_result(diag, sub_diag, eigenvalue, eigenvector):
    """
    验证特征值和特征向量的正确性
    
    参数:
    diag: 主对角线
    sub_diag: 次对角线  
    eigenvalue: 特征值
    eigenvector: 特征向量
    
    返回:
    残差范数
    """
    n = len(diag)
    
    # 计算 A * v
    Av = np.zeros(n)
    
    # 第一行
    Av[0] = diag[0] * eigenvector[0] + sub_diag[0] * eigenvector[1]
    
    # 中间行
    for i in range(1, n-1):
        Av[i] = sub_diag[i-1] * eigenvector[i-1] + diag[i] * eigenvector[i] + sub_diag[i] * eigenvector[i+1]
    
    # 最后一行
    Av[n-1] = sub_diag[n-2] * eigenvector[n-2] + diag[n-1] * eigenvector[n-1]
    
    # 计算残差 ||Av - λv||
    residual = Av - eigenvalue * eigenvector
    return np.linalg.norm(residual)

def main():
    """
    主函数：计算100阶测试矩阵的最大、最小特征值和对应特征向量
    """
    print("对称三对角矩阵特征值求解器")
    print("=" * 50)
    
    # 创建100阶测试矩阵
    n = 100
    diag, sub_diag = create_test_matrix(n)
    
    print(f"矩阵规模: {n} × {n}")
    print("矩阵描述: 主对角线为2，上下次对角线为-1")
    print()
    
    # 求最小特征值（第1个）
    print("计算最小特征值...")
    min_eigenvalue = bisection_eigenvalue(diag, sub_diag, 1)
    print(f"最小特征值: {min_eigenvalue:.12f}")
    
    # 求最小特征值对应的特征向量
    print("计算最小特征值对应的特征向量...")
    min_eigenvector = inverse_power_method(diag, sub_diag, min_eigenvalue)
    
    # 验证最小特征值结果
    min_residual = verify_result(diag, sub_diag, min_eigenvalue, min_eigenvector)
    print(f"最小特征值验证残差: {min_residual:.2e}")
    print()
    
    # 求最大特征值（第n个）
    print("计算最大特征值...")
    max_eigenvalue = bisection_eigenvalue(diag, sub_diag, n)
    print(f"最大特征值: {max_eigenvalue:.12f}")
    
    # 求最大特征值对应的特征向量
    print("计算最大特征值对应的特征向量...")
    max_eigenvector = inverse_power_method(diag, sub_diag, max_eigenvalue)
    
    # 验证最大特征值结果
    max_residual = verify_result(diag, sub_diag, max_eigenvalue, max_eigenvector)
    print(f"最大特征值验证残差: {max_residual:.2e}")
    print()
    
    # 理论值比较（对于这个特定矩阵，特征值有解析解）
    print("理论值比较:")
    theoretical_eigenvalues = []
    for k in range(1, n+1):
        lambda_k = 2 - 2 * np.cos(k * np.pi / (n + 1))
        theoretical_eigenvalues.append(lambda_k)
    
    theoretical_min = min(theoretical_eigenvalues)
    theoretical_max = max(theoretical_eigenvalues)
    
    print(f"理论最小特征值: {theoretical_min:.12f}")
    print(f"理论最大特征值: {theoretical_max:.12f}")
    print(f"最小特征值误差: {abs(min_eigenvalue - theoretical_min):.2e}")
    print(f"最大特征值误差: {abs(max_eigenvalue - theoretical_max):.2e}")
    print()
    
    # 显示特征向量的前几个分量
    print("特征向量展示（前10个分量）:")
    print("最小特征值对应特征向量:")
    for i in range(min(10, len(min_eigenvector))):
        print(f"  v[{i}] = {min_eigenvector[i]:10.6f}")
    
    print("\n最大特征值对应特征向量:")
    for i in range(min(10, len(max_eigenvector))):
        print(f"  v[{i}] = {max_eigenvector[i]:10.6f}")
    

    
    return min_eigenvalue, min_eigenvector, max_eigenvalue, max_eigenvector

if __name__ == "__main__":
    min_eigenvalue, min_eigenvector, max_eigenvalue, max_eigenvector = main()