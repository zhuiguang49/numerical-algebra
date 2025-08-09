import numpy as np
from householder import householder_vector

def qr_decomposition(A):
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)
    householders = []

    for k in range(n):
        x = R[k:, k]
        if np.linalg.norm(x) < 1e-10:
            continue
            
        v, beta = householder_vector(x)
        v = v.reshape(-1, 1)
        
        R[k:, k:] -= beta * (v @ (v.T @ R[k:, k:]))
        
        householders.append( (v.flatten(), beta, k) )

    for v, beta, k in reversed(householders):
        v = v.reshape(-1, 1)
        Q[k:, :] -= beta * v @ (v.T @ Q[k:, :])

    return Q, R

def solve_linear_system(A, b, lambda_reg=1e-8):
    Q, R = qr_decomposition(A)
    y = Q.T @ b
    n = A.shape[1]
    
    R_top = R[:n, :]
    R_reg = R_top.T @ R_top + lambda_reg * np.eye(n)
    
    try:
        return np.linalg.solve(R_reg, R_top.T @ y[:n])
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(R_reg, R_top.T @ y[:n], rcond=None)[0]

def least_squares(A, b):
    Q, R = qr_decomposition(A)
    return np.linalg.solve(R[:A.shape[1], :A.shape[1]], (Q.T @ b)[:A.shape[1]])

def matrix_rank(R, tol=1e-10):
    return np.sum(np.abs(np.diag(R)) > tol)