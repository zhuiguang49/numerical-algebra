import numpy as np

def householder_vector(x):
    x = x.astype(float)
    n = x.shape[0]
    
    x = x.reshape(-1, 1) if x.ndim == 1 else x

    sigma = np.linalg.norm(x[1:])**2
    v = np.zeros_like(x)
    v[0] = x[0] + np.sign(x[0])*np.sqrt(x[0]**2 + sigma)
    v[1:] = x[1:]
    
    if np.linalg.norm(v) < 1e-10:
        beta = 0.0
    else:
        beta = 2.0 / (v.T @ v)
    
    return v.flatten(), beta.item()

def householder_matrix(v, beta):
    n = v.shape[0]
    v = v.reshape(-1, 1)
    H = np.eye(n) - beta * (v @ v.T)
    return H

