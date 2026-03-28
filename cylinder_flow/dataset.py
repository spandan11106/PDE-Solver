import torch
import numpy as np
from scipy.stats import qmc 

def generate_points(n_colloc=5000, n_bnd=500, T_max=5.0):
    # Domain bounds (standard cylinder wake setup)
    t_min, t_max = 0.0, T_max
    x_min, x_max = -1.0, 5.0
    y_min, y_max = -2.0, 2.0  
    
    # Cylinder geometry
    cyl_r = 0.5
    cyl_x, cyl_y = 0.0, 0.0

    # 1. Spatiotemporal Collocation Points
    sampler = qmc.LatinHypercube(d=3)
    lhs_samples = sampler.random(n=n_colloc * 2) # Oversample to account for cylinder cutout
    
    t_c = t_min + (t_max - t_min) * lhs_samples[:, 0:1]
    x_c = x_min + (x_max - x_min) * lhs_samples[:, 1:2]
    y_c = y_min + (y_max - y_min) * lhs_samples[:, 2:3]
    
    # Filter out points inside the cylinder
    mask = ((x_c - cyl_x)**2 + (y_c - cyl_y)**2) >= cyl_r**2
    t_c, x_c, y_c = t_c[mask].reshape(-1, 1), x_c[mask].reshape(-1, 1), y_c[mask].reshape(-1, 1)
    
    # Keep only required amount
    t_c, x_c, y_c = t_c[:n_colloc], x_c[:n_colloc], y_c[:n_colloc]
    X_colloc = torch.tensor(np.hstack([t_c, x_c, y_c]), dtype=torch.float32)

    # 2. Boundary Conditions (Inlet, Cylinder Wall, Initial Condition)
    sampler_2d = qmc.LatinHypercube(d=2)
    
    # A. Inlet Boundary (x = x_min) -> u=1, v=0
    inlet_samples = sampler_2d.random(n=n_bnd)
    t_in = t_min + (t_max - t_min) * inlet_samples[:, 0:1]
    y_in = y_min + (y_max - y_min) * inlet_samples[:, 1:2]
    x_in = np.full((n_bnd, 1), x_min)
    X_inlet = np.hstack([t_in, x_in, y_in])
    U_inlet = np.hstack([np.ones((n_bnd, 1)), np.zeros((n_bnd, 1)), np.zeros((n_bnd, 1))]) # u,v,p (p is dummy here)

    # B. Cylinder Surface (No-slip) -> u=0, v=0
    cyl_samples = sampler_2d.random(n=n_bnd)
    t_cyl = t_min + (t_max - t_min) * cyl_samples[:, 0:1]
    theta = 2 * np.pi * cyl_samples[:, 1:2]
    x_cyl = cyl_x + cyl_r * np.cos(theta)
    y_cyl = cyl_y + cyl_r * np.sin(theta)
    X_cyl = np.hstack([t_cyl, x_cyl, y_cyl])
    U_cyl = np.zeros((n_bnd, 3)) # u=0, v=0, p=0 (p is dummy)

    # Aggregate Boundaries
    X_bnd = torch.tensor(np.vstack([X_inlet, X_cyl]), dtype=torch.float32)
    
    # Targets (We only supervise u and v for inlet and cylinder. We ignore p MSE for these)
    # Target shape: [u, v, mask_u, mask_v] so we know what to compute loss against
    # For simplicity, we'll just return u and v arrays.
    u_bnd = torch.tensor(np.vstack([U_inlet[:, 0:1], U_cyl[:, 0:1]]), dtype=torch.float32)
    v_bnd = torch.tensor(np.vstack([U_inlet[:, 1:2], U_cyl[:, 1:2]]), dtype=torch.float32)

    return X_colloc, X_bnd, u_bnd, v_bnd