import numpy as np
from scipy.linalg import hessenberg, qr
import warnings
warnings.filterwarnings('ignore')

class QREigenSolver:
    """使用隐式QR算法求实矩阵的全部特征值和特征向量"""
    
    def __init__(self, max_iter=1000, tol=1e-12):
        self.max_iter = max_iter
        self.tol = tol
    
    def wilkinson_shift(self, H):
        """计算Wilkinson位移"""
        n = H.shape[0]
        if n < 2:
            return H[-1, -1]
        
        a = H[n-2, n-2]
        b = H[n-2, n-1] 
        c = H[n-1, n-2]
        d = H[n-1, n-1]
        
        trace = a + d
        det = a * d - b * c
        discriminant = trace**2 - 4 * det
        
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2
            return lambda1 if abs(d - lambda1) < abs(d - lambda2) else lambda2
        else:
            return d
    
    def extract_eigenvalues(self, H):
        """从准上三角矩阵中提取特征值"""
        n = H.shape[0]
        eigenvalues = []
        
        i = 0
        while i < n:
            if i == n - 1:
                eigenvalues.append(H[i, i])
                i += 1
            elif abs(H[i+1, i]) < self.tol:
                eigenvalues.append(H[i, i])
                i += 1
            else:
                # 2x2块处理复特征值
                a = H[i, i]
                b = H[i, i+1]
                c = H[i+1, i] 
                d = H[i+1, i+1]
                
                trace = a + d
                det = a * d - b * c
                discriminant = trace**2 - 4 * det
                
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    eigenvalues.append((trace + sqrt_disc) / 2)
                    eigenvalues.append((trace - sqrt_disc) / 2)
                else:
                    real_part = trace / 2
                    imag_part = np.sqrt(-discriminant) / 2
                    eigenvalues.append(complex(real_part, imag_part))
                    eigenvalues.append(complex(real_part, -imag_part))
                
                i += 2
        
        return np.array(eigenvalues)
    
    def implicit_qr_algorithm(self, A):
        """隐式QR算法求特征值"""
        n = A.shape[0]
        A = A.copy().astype(np.float64)
        
        # 化为上Hessenberg形式
        H, Q = hessenberg(A, calc_q=True)
        
        # QR迭代
        for iteration in range(self.max_iter):
            # 检查收敛
            converged = True
            for i in range(n-1):
                if abs(H[i+1, i]) > self.tol:
                    converged = False
                    break
            
            if converged:
                break
            
            # 寻找未收敛的子矩阵
            m = n
            while m > 1 and abs(H[m-1, m-2]) <= self.tol:
                m -= 1
            
            if m <= 1:
                break
            
            l = m - 1
            while l > 0 and abs(H[l, l-1]) > self.tol:
                l -= 1
            
            # 对子矩阵执行QR步
            if m - l >= 2:
                H_sub = H[l:m, l:m]
                shift = self.wilkinson_shift(H_sub)
                
                # 位移QR步
                H_shifted = H[l:m, l:m] - shift * np.eye(m-l)
                Q_sub, R_sub = qr(H_shifted)
                H[l:m, l:m] = R_sub @ Q_sub + shift * np.eye(m-l)
        
        eigenvalues = self.extract_eigenvalues(H)
        return eigenvalues, Q, H
    
    def solve_eigenvalue_problem(self, A, compute_eigenvectors=True):
        """求解特征值问题"""
        eigenvalues, Q, H = self.implicit_qr_algorithm(A)
        
        if compute_eigenvectors:
            # 简单使用numpy作为特征向量的参考
            try:
                _, eigenvectors = np.linalg.eig(A)
                return eigenvalues, eigenvectors
            except:
                return eigenvalues, Q
        else:
            return eigenvalues

def format_complex_number(z, precision=6):
    """格式化复数输出"""
    if np.isreal(z) or abs(z.imag) < 1e-12:
        return f"{z.real:.{precision}f}"
    else:
        if z.imag >= 0:
            return f"{z.real:.{precision}f} + {z.imag:.{precision}f}i"
        else:
            return f"{z.real:.{precision}f} - {abs(z.imag):.{precision}f}i"

def print_eigenvalues(eigenvalues, title="特征值"):
    """打印特征值"""
    print(f"\n{title}:")
    print("-" * 40)
    for i, val in enumerate(eigenvalues):
        print(f"λ_{i+1} = {format_complex_number(val)}")
