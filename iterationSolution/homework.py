import numpy as np
from jacobi import jacobi
from gs import gauss_seidel
from sor import sor

a = 0.5
n = 100
h = 1/n

def build_system(epsilon):
    """构建线性方程组 AX = b"""
    n_nodes = n-1  # 内部节点数
    main_diag = np.full(n_nodes, -(2*epsilon + h))
    lower_diag = np.full(n_nodes-1, epsilon)
    upper_diag = np.full(n_nodes-1, epsilon + h)
    
    # 三对角矩阵
    A = np.diag(main_diag) + np.diag(lower_diag, -1) + np.diag(upper_diag, 1)
    
    b = np.full(n_nodes, a*h**2)
    b[-1] -= (epsilon + h) * 1  
    
    return A, b

def exact_solution(x, epsilon):
    """精确解计算公式"""
    return ((1-a)/(1 - np.exp(-1/epsilon))) * (1 - np.exp(-x/epsilon)) + a*x

# 公共参数设置
max_iter = 10000
tol = 1e-8  # 确保4位有效数字


def solve_jacobi(epsilon):
    A, b = build_system(epsilon)
    x = jacobi(A, b, max_iter=max_iter, tol=tol)
    return x

def solve_gs(epsilon):
    A, b = build_system(epsilon)
    x = gauss_seidel(A, b, max_iter=max_iter, tol=tol)
    return x

def solve_sor(epsilon):
    A, b = build_system(epsilon)
    omega = 1.2
    x = sor(A, b, omega, max_iter=max_iter, tol=tol)
    return x

def format_value(val):
    """格式化为四位有效数字的科学计数法"""
    return f"{val:.4e}"

if __name__ == "__main__":
    test_epsilons = [1, 0.1, 0.01, 0.0001]
    sample_indices = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]  # 前5和后5个节点
    
    for eps in test_epsilons:
        print(f"\nε = {eps}:")
        A, b = build_system(eps)
        x_jacobi = solve_jacobi(eps)
        x_gs = solve_gs(eps)
        x_sor = solve_sor(eps)
        x_points = np.linspace(0, 1, n+1)[1:-1]
        exact = exact_solution(x_points, eps)
        
        # 输出部分节点解
        print("节点位置\tJacobi解\tGauss-Seidel解\tSOR解\t\t精确解")
        for idx in sample_indices:
            xi = x_points[idx]
            j_val = format_value(x_jacobi[idx])
            gs_val = format_value(x_gs[idx])
            sor_val = format_value(x_sor[idx])
            ex_val = format_value(exact[idx])
            print(f"{xi:.4f}\t{j_val}\t{gs_val}\t{sor_val}\t{ex_val}")
        
        # 计算并输出最大误差
        errors = {
            'Jacobi': np.max(np.abs(x_jacobi - exact)),
            'Gauss-Seidel': np.max(np.abs(x_gs - exact)),
            'SOR': np.max(np.abs(x_sor - exact))
        }
        print("\n最大误差（四位有效数字）:")
        for method, error in errors.items():
            print(f"{method}: {format_value(error)}")