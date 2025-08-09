import numpy as np
import optimization
import forward_substitution
import backward_substitution
import gaussian_elimination_column

def generate_matrix(n):
    '''
    用于生成矩阵
    '''
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = 1.0
        A[i, n-1] = 1.0
        for j in range(i):
            A[i, j] = -1.0
    return A

if __name__ == "__main__":
    # 用于保存输出结果的字符串列表
    output_lines = []
    header = "  n |   cond(A)   |  相对残量      |  预测前向误差上界       |  实际相对误差\n"
    separator = "-" * len(header) + "\n"
    output_lines.append(header)
    output_lines.append(separator)
    
    # 对 n 从 5 到 30 循环
    for n in range(5, 31):
        # 1. 生成矩阵 A
        A = generate_matrix(n)
        
        # 2. 随机生成真实解 x_true
        x_true = np.random.rand(n)
        
        # 3. 计算右端项 b = A * x_true
        b = A @ x_true
        
        # 4. 用列主元 Gauss 消去法求解，得到计算解 x_approx
        x_approx, L, U, P = gaussian_elimination_column.gaussian_elimination(A, b)
        
        # 5. 计算实际的相对误差（无穷范数）
        error_norm = np.linalg.norm(x_approx - x_true, ord=np.inf)
        true_norm = np.linalg.norm(x_true, ord=np.inf)
        rel_error = error_norm / true_norm
        
        # 6. 计算残量 r = b - A * x_approx，及其相对残量
        r = b - A @ x_approx
        rel_res = np.linalg.norm(r, ord=np.inf) / np.linalg.norm(b, ord=np.inf)
        
        # 7. 利用盲人爬山法估计 A 的 1 范数和 A^{-1} 的 1 范数
        norm_A, _ = optimization.estimate_matrix_1_norm(A)
        invA = np.linalg.inv(A)
        norm_invA, _ = optimization.estimate_matrix_1_norm(invA)
        cond_est = norm_A * norm_invA  # cond_1(A) 的估计
        
        # 8. 依据理论，前向误差上界可估计为： cond(A) * (相对残量)
        pred_error_bound = cond_est * rel_res
        
        # 9. 格式化输出，每一行包含：n、估计的条件数、相对残量、预测前向误差上界、实际相对误差
        line = f"{n:3d} | {cond_est:10.4e} | {rel_res:14.4e} | {pred_error_bound:22.4e} | {rel_error:18.4e}\n"
        output_lines.append(line)
    
    # 10. 将所有结果写入 output.txt 文件中
    with open("output.txt", "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    
    print("结果已写入 output.txt 文件中。")
