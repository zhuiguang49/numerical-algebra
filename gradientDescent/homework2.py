import numpy as np
from steepestDescent import steepest_descent
from conjugateGradient import conjugate_gradient
from gs import gauss_seidel
from jacobi import jacobi
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
except:
    # 如果找不到指定字体，尝试使用系统默认中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong']
    plt.rcParams['axes.unicode_minus'] = False

A = np.array([[10,1,2,3,4],
              [1,9,-1,2,-3],
              [2,-1,7,3,-5],
              [3,2,3,12,-1],
              [4,-3,-5,-1,15]], dtype=float)
b = np.array([12,-27,14,-17,12], dtype=float)

# 精确解参考
x_exact = np.linalg.solve(A, b)

# Jacobi迭代法测试
result_jacobi = jacobi(A, b, max_iter=1000, tol=1e-6)
error_jacobi = np.linalg.norm(result_jacobi['x'] - x_exact)

# Gauss-Seidel迭代法测试
result_gs = gauss_seidel(A, b, max_iter=1000, tol=1e-6)
error_gs = np.linalg.norm(result_gs['x'] - x_exact)

# 共轭梯度法测试
result_cg = conjugate_gradient(A, b, x0=np.zeros(5), max_iter=1000, tol=1e-6)
error_cg = np.linalg.norm(result_cg['x'] - x_exact)

# 结果对比
print("="*50)
print(f"{'方法':<15} | {'误差':<10} | {'迭代次数':<8}")
print("-"*50)
print(f"{'Jacobi':<15} | {error_jacobi:.2e} | {result_jacobi['iterations']}")
print(f"{'Gauss-Seidel':<15} | {error_gs:.2e} | {result_gs['iterations']}")
print(f"{'Conjugate Gradient':<15} | {error_cg:.2e} | {result_cg['iterations']}")

# 打印解向量
print("\n各方法解向量:")
print("-"*50)
print(f"{'方法':<15} | {'x[0]':<10} | {'x[1]':<10} | {'x[2]':<10} | {'x[3]':<10} | {'x[4]':<10}")
print("-"*50)
print(f"{'精确解':<15} | {x_exact[0]:10.6f} | {x_exact[1]:10.6f} | {x_exact[2]:10.6f} | {x_exact[3]:10.6f} | {x_exact[4]:10.6f}")
print(f"{'Jacobi':<15} | {result_jacobi['x'][0]:10.6f} | {result_jacobi['x'][1]:10.6f} | {result_jacobi['x'][2]:10.6f} | {result_jacobi['x'][3]:10.6f} | {result_jacobi['x'][4]:10.6f}")
print(f"{'Gauss-Seidel':<15} | {result_gs['x'][0]:10.6f} | {result_gs['x'][1]:10.6f} | {result_gs['x'][2]:10.6f} | {result_gs['x'][3]:10.6f} | {result_gs['x'][4]:10.6f}")
print(f"{'Conjugate Gradient':<15} | {result_cg['x'][0]:10.6f} | {result_cg['x'][1]:10.6f} | {result_cg['x'][2]:10.6f} | {result_cg['x'][3]:10.6f} | {result_cg['x'][4]:10.6f}")

# 残差下降可视化
plt.figure(figsize=(10,6))
plt.semilogy(result_jacobi['residuals'], label='Jacobi')
plt.semilogy(result_gs['residuals'], label='Gauss-Seidel')
plt.semilogy(result_cg['residuals'], label='Conjugate Gradient')
plt.title('残差收敛对比')
plt.xlabel('迭代次数')
plt.ylabel('残差范数 (log尺度)')
plt.legend()
plt.grid(True)
plt.show()

# 另外绘制一个迭代收敛前20次的细节图
plt.figure(figsize=(10,6))
max_iter_to_show = min(20, 
                      max(len(result_jacobi['residuals']), 
                          len(result_gs['residuals']), 
                          len(result_cg['residuals'])))
plt.semilogy(result_jacobi['residuals'][:max_iter_to_show], 'o-', label='Jacobi')
plt.semilogy(result_gs['residuals'][:max_iter_to_show], 's-', label='Gauss-Seidel')
plt.semilogy(result_cg['residuals'][:max_iter_to_show], '^-', label='Conjugate Gradient')
plt.title('前20次迭代残差收敛对比')
plt.xlabel('迭代次数')
plt.ylabel('残差范数 (log尺度)')
plt.legend()
plt.grid(True)
plt.show()