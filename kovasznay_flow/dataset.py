import torch
import numpy as np
from scipy.stats import qmc 
from kovasznay import kovasznay_solution

def generate_points(n_colloc = 5000, n_bnd = 200):
    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 0.5  

    sampler = qmc.LatinHypercube(d=2)

    lhs_samples = sampler.random(n=n_colloc)

    x_c = x_min + (x_max - x_min) * lhs_samples[:, 0:1]
    y_c = y_min + (y_max - y_min) * lhs_samples[:, 1:2]

    X_colloc = torch.tensor(np.hstack([x_c, y_c]), dtype = torch.float32)

    sampler_1d = qmc.LatinHypercube(d=1)
    
    # Left Boundary (x = x_min)
    y_left_samples = sampler_1d.random(n=n_bnd)
    y_left = y_min + (y_max - y_min) * y_left_samples
    x_left = np.full((n_bnd, 1), x_min)
    X_left = torch.tensor(np.hstack([x_left, y_left]), dtype=torch.float32)
    
    # Right Boundary (x = x_max)
    y_right_samples = sampler_1d.random(n=n_bnd)
    y_right = y_min + (y_max - y_min) * y_right_samples
    x_right = np.full((n_bnd, 1), x_max)
    X_right = torch.tensor(np.hstack([x_right, y_right]), dtype=torch.float32)

    # Top Boundary (y = y_max)
    x_top_samples = sampler_1d.random(n=n_bnd)
    x_top = x_min + (x_max - x_min) * x_top_samples
    y_top = np.full((n_bnd, 1), y_max)
    X_top = torch.tensor(np.hstack([x_top, y_top]), dtype=torch.float32)

    # Bottom Boundary (y = y_min)
    x_bottom_samples = sampler_1d.random(n=n_bnd)
    x_bottom = x_min + (x_max - x_min) * x_bottom_samples
    y_bottom = np.full((n_bnd, 1), y_min)
    X_bottom = torch.tensor(np.hstack([x_bottom, y_bottom]), dtype=torch.float32)

    X_bnd = torch.cat([X_left, X_right, X_top, X_bottom], dim=0)
    u_bnd, v_bnd, p_bnd = kovasznay_solution(X_bnd[:, 0:1], X_bnd[:, 1:2])

    return X_colloc, X_bnd, u_bnd, v_bnd, p_bnd