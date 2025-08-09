import numpy as np

def power_method_for_polynomial(coefficients, max_iter=1000, tol=1e-6):
    n = len(coefficients) - 1  
    if n < 1:
        raise ValueError("多项式次数必须至少为1")

    C = np.zeros((n, n))
    for i in range(n - 1):
        C[i + 1, i] = 1
    C[:, -1] = -np.array(coefficients[:-1]) / coefficients[-1]

    v = np.ones(n)
    eigenvalue = 0

    for _ in range(max_iter):
        v_new = np.dot(C, v)
        v_new_norm = np.linalg.norm(v_new, np.inf)
        v_new = v_new / v_new_norm
        eigenvalue_new = np.dot(v_new, np.dot(C, v_new)) / np.dot(v_new, v_new)
        
        if np.abs(eigenvalue_new - eigenvalue) < tol:
            break
        
        v = v_new
        eigenvalue = eigenvalue_new

    return eigenvalue, v