import numpy as np
from qr_eigenvalue import QREigenSolver, format_complex_number
import warnings
warnings.filterwarnings('ignore')

def create_matrix_A(x):
    """创建参数为x的矩阵A"""
    A = np.array([
        [9.1, 3.0, 2.6, 4.0],
        [4.2, 5.3, 4.7, 1.6],
        [3.2, 1.7, 9.4, x  ],
        [6.1, 4.9, 3.5, 6.2]
    ], dtype=float)
    return A

def main():
    """分析矩阵A在x=0.9,1.0,1.1时的特征值变化"""
    print("=" * 60)
    print("矩阵特征值敏感性分析")
    print("=" * 60)
    print("矩阵A:")
    print("[9.1  3.0  2.6  4.0]")
    print("[4.2  5.3  4.7  1.6]")
    print("[3.2  1.7  9.4   x ]")
    print("[6.1  4.9  3.5  6.2]")
    
    solver = QREigenSolver(max_iter=1000, tol=1e-12)
    x_values = [0.9, 1.0, 1.1]
    results = {}
    
    # 计算各x值对应的特征值
    for x in x_values:
        print(f"\n当 x = {x} 时:")
        print("-" * 30)
        
        A = create_matrix_A(x)
        eigenvalues = solver.solve_eigenvalue_problem(A, compute_eigenvectors=False)
        
        # 排序特征值
        eigenvalues_sorted = sorted(eigenvalues, key=lambda v: v.real)
        
        print("特征值:")
        for i, val in enumerate(eigenvalues_sorted):
            print(f"  λ_{i+1} = {format_complex_number(val, precision=8)}")
        
        # 计算矩阵统计量
        trace_computed = sum(val.real for val in eigenvalues_sorted)
        trace_theory = np.trace(A)
        det_theory = np.linalg.det(A)
        
        print(f"矩阵的迹: {trace_computed:.8f} (理论值: {trace_theory:.8f})")
        print(f"行列式: {det_theory:.8f}")
        
        results[x] = eigenvalues_sorted
    
    # 分析特征值变化
    print(f"\n\n特征值变化分析:")
    print("=" * 60)
    print(f"{'x值':<8} {'特征值1':<18} {'特征值2':<18} {'特征值3':<18} {'特征值4':<18}")
    print("-" * 80)
    
    for x in x_values:
        eigenvals = results[x]
        row = f"{x:<8.1f} "
        for val in eigenvals:
            row += f"{format_complex_number(val, precision=6):<18} "
        print(row)
    
    # 计算特征值变化量
    print(f"\n特征值变化分析:")
    print("-" * 40)
    
    for i in range(4):
        print(f"\n第{i+1}个特征值的变化:")
        val_09 = results[0.9][i]
        val_10 = results[1.0][i]
        val_11 = results[1.1][i]
        
        if np.isreal(val_09) and np.isreal(val_10) and np.isreal(val_11):
            change1 = val_10.real - val_09.real
            change2 = val_11.real - val_10.real
            print(f"  x: 0.9 → 1.0, 变化: {change1:+.8f}")
            print(f"  x: 1.0 → 1.1, 变化: {change2:+.8f}")
            print(f"  平均变化率: {(change1 + change2)/0.2:.8f} per unit")
        else:
            print(f"  x=0.9: {format_complex_number(val_09)}")
            print(f"  x=1.0: {format_complex_number(val_10)}")
            print(f"  x=1.1: {format_complex_number(val_11)}")
    
    # 敏感性分析
    print(f"\n敏感性分析 (相对于x=1.0):")
    print("-" * 40)
    
    eigenvals_ref = results[1.0]
    for x in [0.9, 1.1]:
        eigenvals_x = results[x]
        dx = x - 1.0
        print(f"\n当x={x} (Δx={dx:+.1f})时:")
        
        for i in range(4):
            if np.isreal(eigenvals_ref[i]) and np.isreal(eigenvals_x[i]):
                dlambda = eigenvals_x[i].real - eigenvals_ref[i].real
                sensitivity = dlambda / dx
                print(f"  特征值{i+1}敏感性: dλ/dx ≈ {sensitivity:.6f}")

if __name__ == "__main__":
    main()