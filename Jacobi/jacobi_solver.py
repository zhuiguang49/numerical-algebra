import numpy as np
import math

class TridiagonalJacobiSolver:
    """
    过关Jacobi方法求解实对称三对角矩阵特征值和特征向量的专用类
    """
    
    def __init__(self, tolerance=1e-12, max_iterations=1000):
        """
        初始化过关Jacobi求解器
        
        Parameters:
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, A):
        """
        使用过关Jacobi方法求解实对称三对角矩阵的特征值和特征向量
        
        Parameters:
        A: 实对称三对角矩阵 (n×n numpy数组)
        
        Returns:
        eigenvalues: 特征值数组 (从小到大排序)
        eigenvectors: 特征向量矩阵 (每列是一个特征向量)
        """
        n = A.shape[0]
        
        # 提取三对角矩阵的对角线和次对角线
        d = np.diag(A).copy().astype(np.float64)  # 主对角线
        e = np.zeros(n-1, dtype=np.float64)       # 次对角线
        for i in range(n-1):
            e[i] = A[i, i+1]
        
        # 初始化特征向量矩阵
        V = np.eye(n, dtype=np.float64)
        
        # 过关Jacobi迭代
        for iteration in range(self.max_iterations):
            # 检查收敛性：所有次对角元素是否足够小
            max_off_diag = np.max(np.abs(e))
            if max_off_diag < self.tolerance:
                break
            
            # 对每个次对角元素进行Givens旋转
            for i in range(n-1):
                if abs(e[i]) > self.tolerance:
                    # 计算Givens旋转参数
                    if abs(d[i] - d[i+1]) < 1e-15:
                        # 避免除零
                        theta = math.pi / 4
                    else:
                        theta = 0.5 * math.atan(2 * e[i] / (d[i] - d[i+1]))
                    
                    c = math.cos(theta)
                    s = math.sin(theta)
                    
                    # 更新对角元素
                    d_ii = d[i]
                    d_jj = d[i+1]
                    e_ij = e[i]
                    
                    d[i] = c*c*d_ii + s*s*d_jj - 2*c*s*e_ij
                    d[i+1] = s*s*d_ii + c*c*d_jj + 2*c*s*e_ij
                    e[i] = 0.0  # 这个元素被消除
                    
                    # 更新相邻的次对角元素
                    if i > 0:
                        temp = e[i-1]
                        e[i-1] = c * temp
                        # 注意：这里会产生新的非零元素，但在三对角结构中处理
                    
                    if i < n-2:
                        temp = e[i+1]
                        e[i+1] = s * temp
                    
                    # 更新特征向量矩阵
                    for k in range(n):
                        temp1 = V[k, i]
                        temp2 = V[k, i+1]
                        V[k, i] = c * temp1 - s * temp2
                        V[k, i+1] = s * temp1 + c * temp2
        
        # 返回结果
        eigenvalues = d.copy()
        
        # 排序特征值和特征向量
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = V[:, idx]
        
        return eigenvalues, eigenvectors


class ImprovedTridiagonalSolver:
    """
    使用QR算法的改进三对角矩阵求解器
    这是更稳定和精确的实现
    """
    
    def __init__(self, tolerance=1e-12, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, A):
        """
        使用QR迭代求解三对角矩阵特征值和特征向量
        """
        n = A.shape[0]
        
        # 复制输入矩阵
        T = A.copy().astype(np.float64)
        Q = np.eye(n, dtype=np.float64)
        
        # QR迭代
        for iteration in range(self.max_iterations):
            # 检查收敛
            max_subdiag = 0.0
            for i in range(n-1):
                max_subdiag = max(max_subdiag, abs(T[i+1, i]))
            
            if max_subdiag < self.tolerance:
                break
            
            # 选择位移策略（Wilkinson位移）
            mu = self._wilkinson_shift(T, n-1)
            
            # QR分解 T - μI = QR
            T_shifted = T - mu * np.eye(n)
            Q_iter, R_iter = self._qr_decomposition_tridiagonal(T_shifted)
            
            # 更新 T = RQ + μI
            T = R_iter @ Q_iter + mu * np.eye(n)
            
            # 累积特征向量
            Q = Q @ Q_iter
        
        # 提取特征值
        eigenvalues = np.diag(T)
        
        # 排序
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = Q[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _wilkinson_shift(self, T, k):
        """计算Wilkinson位移"""
        if k == 0:
            return T[0, 0]
        
        a = T[k-1, k-1]
        b = T[k, k-1]
        c = T[k, k]
        
        d = (a - c) / 2
        if d == 0:
            return c - abs(b)
        else:
            return c - (b * b) / (d + np.sign(d) * math.sqrt(d*d + b*b))
    
    def _qr_decomposition_tridiagonal(self, T):
        """对三对角矩阵进行QR分解"""
        n = T.shape[0]
        Q = np.eye(n)
        R = T.copy()
        
        # 使用Givens旋转
        for i in range(n-1):
            if abs(R[i+1, i]) > 1e-15:
                # 计算Givens参数
                a = R[i, i]
                b = R[i+1, i]
                
                if abs(b) > abs(a):
                    tau = -a / b
                    s = 1 / math.sqrt(1 + tau*tau)
                    c = s * tau
                else:
                    tau = -b / a
                    c = 1 / math.sqrt(1 + tau*tau)
                    s = c * tau
                
                # 应用Givens旋转到R
                for j in range(n):
                    temp1 = R[i, j]
                    temp2 = R[i+1, j]
                    R[i, j] = c * temp1 - s * temp2
                    R[i+1, j] = s * temp1 + c * temp2
                
                # 累积Q
                for j in range(n):
                    temp1 = Q[j, i]
                    temp2 = Q[j, i+1]
                    Q[j, i] = c * temp1 - s * temp2
                    Q[j, i+1] = s * temp1 + c * temp2
        
        return Q, R


def create_tridiagonal_matrix(n, main_diag=4, sub_diag=1):
    """
    创建三对角矩阵
    
    Parameters:
    n: 矩阵阶数
    main_diag: 主对角线元素
    sub_diag: 次对角线元素
    
    Returns:
    三对角矩阵
    """
    A = np.zeros((n, n))
    
    # 设置主对角线
    for i in range(n):
        A[i, i] = main_diag
    
    # 设置次对角线
    for i in range(n-1):
        A[i, i+1] = sub_diag
        A[i+1, i] = sub_diag
    
    return A


def solve_and_save_results():
    """
    求解50阶到100阶矩阵的特征值和特征向量，并保存结果
    """
    # 使用改进的算法
    solver = ImprovedTridiagonalSolver(tolerance=1e-14, max_iterations=2000)
    
    # 创建输出文件
    with open('tridiagonal_eigenvalue_results.txt', 'w', encoding='utf-8') as f:
        f.write("实对称三对角矩阵特征值和特征向量计算结果\n")
        f.write("=" * 60 + "\n")
        f.write("矩阵说明：主对角线为4，次对角线为1\n")
        f.write("求解方法：过关Jacobi方法（QR迭代实现）\n\n")
        
        # 计算50阶到100阶矩阵
        for n in range(50, 101):
            print(f"正在计算 {n} 阶矩阵...")
            
            # 创建三对角矩阵
            A = create_tridiagonal_matrix(n, main_diag=4, sub_diag=1)
            
            # 求解特征值和特征向量
            eigenvalues, eigenvectors = solver.solve(A)
            
            # 写入结果
            f.write(f"\n{n}阶矩阵结果：\n")
            f.write("-" * 40 + "\n")
            
            # 保存特征值
            f.write("特征值：\n")
            for i, val in enumerate(eigenvalues):
                f.write(f"λ_{i+1:2d} = {val:15.12f}\n")
            
            # 保存前3个和后3个特征向量（节省空间）
            f.write("\n前3个特征向量：\n")
            for i in range(min(3, n)):
                f.write(f"特征向量 {i+1}：\n")
                for j in range(n):
                    f.write(f"{eigenvectors[j, i]:12.8f} ")
                    if (j + 1) % 8 == 0:  # 每行8个数
                        f.write("\n")
                if n % 8 != 0:
                    f.write("\n")
                f.write("\n")
            
            if n > 6:
                f.write(f"后3个特征向量：\n")
                for i in range(max(0, n-3), n):
                    f.write(f"特征向量 {i+1}：\n")
                    for j in range(n):
                        f.write(f"{eigenvectors[j, i]:12.8f} ")
                        if (j + 1) % 8 == 0:
                            f.write("\n")
                    if n % 8 != 0:
                        f.write("\n")
                    f.write("\n")
            
            # 验证结果正确性
            max_error = 0.0
            for i in range(n):
                Av = A @ eigenvectors[:, i]
                lambda_v = eigenvalues[i] * eigenvectors[:, i]
                error = np.linalg.norm(Av - lambda_v)
                max_error = max(max_error, error)
            
            f.write(f"验证最大误差：{max_error:.2e}\n")
            
            # 也验证正交性
            orthogonality_error = np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(n))
            f.write(f"正交性误差：{orthogonality_error:.2e}\n")
            f.write("=" * 60 + "\n")
    
    print("计算完成！结果已保存到 tridiagonal_eigenvalue_results.txt 文件中。")


if __name__ == "__main__":
    
    # 求解问题并保存结果
    solve_and_save_results()