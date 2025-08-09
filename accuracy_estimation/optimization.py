import numpy as np

def estimate_matrix_1_norm(B, max_iter=100, tol=1e-6):
    """
    估计矩阵 B 的一范数 ||B||_1，基于算法 2.5.1（优化法）。

    参数：
    - B: (n, n) 形状的 NumPy 矩阵
    - max_iter: 最大迭代次数，默认为100
    - tol: 容忍度，用于判断收敛，默认为1e-6

    返回：
    - norm_estimate: 估计的 ||B||_1 值
    - x: 使 ||Bx||_1 最大化的向量（1 范数为 1）
    """
    # 输入检查
    if len(B.shape) != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("矩阵 B 必须是方阵")
    n = B.shape[1]

    # 初始化 x，1 范数为 1
    x = np.ones(n) / n  # 初始化为均匀分布

    iter_count = 0
    converged = False

    while iter_count < max_iter and not converged:
        iter_count += 1

        # 计算 w = Bx
        w = B @ x

        # 计算 v = sign(w)
        v = np.sign(w)

        # 计算 z = B^T v
        z = B.T @ v

        # 检查终止条件：||z||_infty <= z^T x + tol
        z_inf_norm = np.max(np.abs(z))
        zTx = np.dot(z, x)

        if z_inf_norm <= zTx + tol:
            converged = True
        else:
            # 找到使得 |z_j| 最大的索引 j
            j = np.argmax(np.abs(z))
            # 更新 x 为 e_j * sign(z_j)，保证 ||x||_1 = 1
            x = np.zeros(n)
            x[j] = np.sign(z[j])

    # 计算最终的 ||Bx||_1 作为估计值
    norm_estimate = np.linalg.norm(B @ x, ord=1)

    return norm_estimate, x