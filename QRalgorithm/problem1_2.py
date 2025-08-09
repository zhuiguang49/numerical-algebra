import numpy as np
from scipy.linalg import eigvals, companion
import warnings
warnings.filterwarnings('ignore')

def format_complex_number(z, precision=6):
    """格式化复数输出"""
    if np.isreal(z) or abs(z.imag) < 1e-12:
        return f"{z.real:.{precision}f}"
    else:
        if z.imag >= 0:
            return f"{z.real:.{precision}f} + {z.imag:.{precision}f}i"
        else:
            return f"{z.real:.{precision}f} - {abs(z.imag):.{precision}f}i"

def create_companion_matrix_correct(coefficients):
    """正确构造多项式的伴随矩阵
    对于多项式 a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0 = 0
    """
    coefficients = np.array(coefficients, dtype=complex)
    n = len(coefficients) - 1
    
    if n <= 0:
        raise ValueError("多项式次数必须大于0")
    if coefficients[0] == 0:
        raise ValueError("首项系数不能为0")
    
    # 构造伴随矩阵 - 使用标准形式
    # 对于 x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0 = 0
    companion_mat = np.zeros((n, n), dtype=complex)
    
    # 上方的单位矩阵部分
    for i in range(n-1):
        companion_mat[i, i+1] = 1
    
    # 最后一行是负的系数比值
    for j in range(n):
        companion_mat[n-1, j] = -coefficients[n-j] / coefficients[0]
    
    return companion_mat

def solve_polynomial_equation_correct(coefficients):
    """正确求解多项式方程的根"""
    # 使用scipy的companion矩阵函数作为对比
    coeffs_normalized = np.array(coefficients, dtype=complex)
    coeffs_normalized = coeffs_normalized / coeffs_normalized[0]  # 归一化首项系数
    
    # 构造伴随矩阵
    companion_mat = create_companion_matrix_correct(coefficients)
    
    # 计算特征值
    eigenvalues = eigvals(companion_mat)
    
    return eigenvalues, companion_mat

def verify_roots_correct(coefficients, roots, tol=1e-8):
    """验证根的正确性"""
    errors = []
    for root in roots:
        # 计算 p(root)
        value = 0
        for i, coeff in enumerate(coefficients):
            power = len(coefficients) - 1 - i
            value += coeff * (root ** power)
        errors.append(abs(value))
    return np.array(errors)

def analyze_polynomial_x41_x3_1():
    """分析多项式 x^41 + x^3 + 1 = 0"""
    print("=" * 70)
    print("多项式方程分析: x^41 + x^3 + 1 = 0")
    print("=" * 70)
    
    # 多项式系数: x^41 + 0*x^40 + ... + 0*x^4 + x^3 + 0*x^2 + 0*x + 1
    coefficients = np.zeros(42)
    coefficients[0] = 1    # x^41 的系数
    coefficients[38] = 1   # x^3 的系数 (位置: 41-3 = 38)
    coefficients[41] = 1   # 常数项
    
    print("多项式系数构造:")
    print(f"  x^41 系数: {coefficients[0]}")
    print(f"  x^3 系数: {coefficients[38]}")  
    print(f"  常数项: {coefficients[41]}")
    print(f"  总次数: {len(coefficients)-1}")
    
    try:
        roots, companion_mat = solve_polynomial_equation_correct(coefficients)
        
        print(f"\n成功找到 {len(roots)} 个根:")
        print("-" * 60)
        
        # 按模长和辐角排序
        roots_sorted = sorted(roots, key=lambda x: (abs(x), np.angle(x)))
        
        for i, root in enumerate(roots_sorted):
            modulus = abs(root)
            angle = np.angle(root) * 180 / np.pi
            print(f"根 {i+1:2d}: {format_complex_number(root, precision=8)} "
                  f"(|z|={modulus:.6f}, arg={angle:6.1f}°)")
        
        # 验证根的正确性
        print(f"\n验证结果:")
        print("-" * 30)
        errors = verify_roots_correct(coefficients, roots)
        print(f"最大误差: {np.max(errors):.2e}")
        print(f"平均误差: {np.mean(errors):.2e}")
        print(f"RMS误差: {np.sqrt(np.mean(errors**2)):.2e}")
        
        good_roots = np.sum(errors < 1e-10)
        print(f"高精度根数量 (误差<1e-10): {good_roots}/{len(roots)}")
        
        # 分析根的分布
        print(f"\n根的统计分析:")
        print("-" * 30)
        real_roots = [z for z in roots if abs(z.imag) < 1e-10]
        complex_roots = [z for z in roots if abs(z.imag) >= 1e-10]
        
        print(f"实根数量: {len(real_roots)}")
        print(f"复根数量: {len(complex_roots)}")
        
        if len(real_roots) > 0:
            print("实根:")
            for root in real_roots:
                print(f"  {root.real:.8f}")
        
        moduli = [abs(z) for z in roots]
        print(f"根的模长范围: [{min(moduli):.6f}, {max(moduli):.6f}]")
        print(f"平均模长: {np.mean(moduli):.6f}")
        
        # 单位圆上的根
        unit_circle_roots = [z for z in roots if abs(abs(z) - 1.0) < 1e-6]
        print(f"近似在单位圆上的根: {len(unit_circle_roots)}")
        
        return roots_sorted, errors
        
    except Exception as e:
        print(f"求解出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_with_theoretical():
    """与理论结果对比"""
    print("\n" + "=" * 70)
    print("理论分析")
    print("=" * 70)
    
    print("对于方程 x^41 + x^3 + 1 = 0:")
    print("1. 这是一个41次多项式，应该有41个复根")
    print("2. 由于所有系数都是实数，复根应成对出现")
    print("3. 可以写成 x^41 = -x^3 - 1")
    print("4. 当 |x| = 1 时，|x^41| = |x^3 + 1|，可能存在单位圆上的根")
    
    # 简单的数值验证
    print("\n手工验证几个特殊值:")
    test_values = [-1, 1, -1j, 1j]
    for val in test_values:
        result = val**41 + val**3 + 1
        print(f"  x = {val}: f(x) = {result}")

if __name__ == "__main__":
    roots, errors = analyze_polynomial_x41_x3_1()
    compare_with_theoretical()
    
    if roots is not None:
        print(f"\n总结: 成功求解了 x^41 + x^3 + 1 = 0 的全部41个根")
        print(f"数值精度: {len([e for e in errors if e < 1e-10])}/41 个根达到1e-10精度")