import numpy as np
from steepestDescent import steepest_descent
from conjugateGradient import conjugate_gradient
from matplotlib import pyplot as plt

A = np.zeros((40,40),dtype=float)
b = np.zeros(40,dtype=float)
for i in range(40):
    for j in range(40):
        A[i,j] = 1/(i+j+1)
    b[i] = np.sum(A[i,:])/3

# 这是供参考的精确解
x_exact = np.linalg.solve(A, b)

# 运行共轭梯度法
result = conjugate_gradient(A, b, x0=np.zeros(40), max_iter=200, tol=1e-6)

# 计算数值误差
error = np.linalg.norm(result['x'] - x_exact)

# 结果分析
print("="*50)
print(f"迭代次数: {result['iterations']}/200")
print(f"最终残差: {result['residuals'][-1]:.2e}")
print(f"与精确解误差: {error:.2e}")

# 打印解向量
print("\n解向量的前10个分量:")
for i in range(min(10, len(result['x']))):
    print(f"x[{i}] = {result['x'][i]:.6e}")

print("\n精确解的前10个分量:")
for i in range(min(10, len(x_exact))):
    print(f"x_exact[{i}] = {x_exact[i]:.6e}")

# 打印解向量与精确解的差异
print("\n解向量与精确解的差异(前10个分量):")
for i in range(min(10, len(x_exact))):
    print(f"差异[{i}] = {result['x'][i] - x_exact[i]:.6e}")

# 计算条件数
cond_num = np.linalg.cond(A)
print(f"\n矩阵条件数: {cond_num:.2e}")

# 残差下降过程图
plt.figure(figsize=(10,6))
plt.semilogy(result['residuals'], marker='o', markersize=3)
plt.title('Conjugate Gradient Convergence (Hilbert Matrix N=40)')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.grid(True)
plt.show()