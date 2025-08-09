import numpy as np
import optimization
import forward_substitution
import backward_substitution

def hilbert_matrix(n):
    H = np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(n):
            H[i,j] = 1.0/(i+j+1)
    return H

if __name__ == "__main__":
    output_lines = []
    header = "  n | cond_1(H_n)（估计）\n"
    separator = "-" * len(header) + "\n"
    output_lines.append(header)
    output_lines.append(separator)
    
    # 对 n 从 5 到 20 计算希尔伯特矩阵的条件数估计
    for n in range(5, 21):
        H = hilbert_matrix(n)
        
        # 估计 H 的 1 范数
        norm_H, _ = optimization.estimate_matrix_1_norm(H)
        # 显式计算 H 的逆，然后估计其 1 范数
        invH = np.linalg.inv(H)
        norm_invH, _ = optimization.estimate_matrix_1_norm(invH)
        
        cond_H = norm_H * norm_invH
        line = f"{n:3d} | {cond_H:12.6e}\n"
        output_lines.append(line)
    
    # 将所有结果写入 hilbert_output.txt 文件中
    with open("hilbert_output.txt", "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    
    print("Hilbert 矩阵的条件数估计结果已写入 hilbert_output.txt 文件中。")
