import numpy as np
from qrdecomposition import least_squares

t = np.array([-1, -0.75, -0.5, 0, 0.25, 0.5, 0.75])
y = np.array([1.00, 0.8125, 0.75, 1.00, 1.3125, 1.75, 2.3125])

A = np.column_stack([t**2, t, np.ones_like(t)])

coefficients = least_squares(A, y)

a, b, c = coefficients.round(6)

residuals = y - (a*t**2 + b*t + c)
residual_norm = np.linalg.norm(residuals).round(6)

print("二次多项式系数:")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"\n残差范数: {residual_norm}")
print("\n拟合结果对比:")
print(" t\t 实际y\t 拟合y\t 残差")
for ti, yi, fit_y in zip(t, y, (a*t**2 + b*t + c)):
    print(f"{ti:5.2f}\t{yi:6.4f}\t{fit_y:6.4f}\t{(yi - fit_y):+7.4f}")