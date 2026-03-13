import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the directory containing network.py and kovasznay.py to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from network import PINN
from kovasznay import kovasznay_solution

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def plot_physics_residuals(model, X_grid, X, Y, Re=20, save_dir="results"):
    model.eval()
    X_grid.requires_grad_(True)
    
    # Get network predictions
    preds = model(X_grid)
    u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    # Calculate gradients as in loss.py
    grads_u = grad(u, X_grid)
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]

    grads_v = grad(v, X_grid)
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    
    # Calculate residuals using your compute_loss logic (u_x + v_y, etc.)
    res_cont = (u_x + v_y).detach().cpu().numpy().reshape(X.shape[0], X.shape[1])
    
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, np.abs(res_cont), 50, cmap='inferno')
    plt.colorbar(label='Continuity Residual')
    plt.title("Where the Physics is Failing")
    
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "physics_residuals.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved physics residuals plot to {plot_path}")
    plt.close()

def plot_error_distribution(u_pred, u_exact, save_dir="results"):
    error = (u_pred - u_exact).flatten()
    plt.figure(figsize=(7, 5))
    plt.hist(error, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.yscale('log') # Use log scale to see tiny outlier errors
    plt.title("Log-Scale Error Distribution")
    plt.xlabel("u_pred - u_exact")
    plt.ylabel("Frequency (Log)")
    
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "error_distribution.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved error distribution plot to {plot_path}")
    plt.close()

def check_mass_conservation(u_pred_grid, v_pred_grid, dx, dy):
    # Calculate flux through 4 sides of the domain
    left_flux = -np.sum(u_pred_grid[:, 0]) * dy
    right_flux = np.sum(u_pred_grid[:, -1]) * dy
    bottom_flux = -np.sum(v_pred_grid[0, :]) * dx
    top_flux = np.sum(v_pred_grid[-1, :]) * dx
    
    total_net_flux = left_flux + right_flux + bottom_flux + top_flux
    print(f"Net Mass Flux (Should be 0): {total_net_flux:.2e}")

def plot_slices(u_pred, u_exact, x_coords, y_coords, x_slice=0.5, save_dir="results"):
    # Find the index closest to the desired x-slice
    idx = (np.abs(x_coords - x_slice)).argmin()
    
    plt.figure(figsize=(8, 4))
    plt.plot(y_coords, u_exact[:, idx], 'k-', label='Exact', linewidth=2)
    plt.plot(y_coords, u_pred[:, idx], 'r--', label='PINN', linewidth=2)
    plt.xlabel('y-coordinate')
    plt.ylabel('u-velocity')
    plt.title(f'Velocity Profile at x = {x_slice}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs(save_dir, exist_ok=True)
    x_slice_str = str(x_coords[idx]).replace('.', '_')
    plot_path = os.path.join(save_dir, f"velocity_profile_x_{x_slice_str}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved slice plot to {plot_path}")
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Establish project root directory
    project_root = os.path.dirname(current_dir)
    save_directory = os.path.join(project_root, 'results')
    
    # 1. Load the model
    model = PINN().to(device)
    model_path = os.path.join(project_root, 'pinn_kovasznay.pth')
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}.")
        sys.exit(1)
        
    model.eval()
    
    # 2. Create a dense grid for plotting
    x = np.linspace(-0.5, 1.0, 101)
    y = np.linspace(-0.5, 0.5, 101)
    X, Y = np.meshgrid(x, y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    # 3. Get predictions and exact solutions
    with torch.no_grad():
        preds = model(grid_tensor)
        u_pred = preds[:, 0].cpu().numpy().reshape(101, 101)
        v_pred = preds[:, 1].cpu().numpy().reshape(101, 101)
        
    x_ex = torch.tensor(grid_points[:, 0:1], dtype=torch.float32)
    y_ex = torch.tensor(grid_points[:, 1:2], dtype=torch.float32)
    u_ex_t, v_ex_t, _ = kovasznay_solution(x_ex, y_ex)
    u_exact = u_ex_t.numpy().reshape(101, 101)
    
    # Run the visualizations
    print(f"Running visualizations and saving to {save_directory}...")
    
    grid_tensor_grad = grid_tensor.clone().detach().requires_grad_(True)
    plot_physics_residuals(model, grid_tensor_grad, X, Y, Re=20, save_dir=save_directory)
    
    plot_error_distribution(u_pred, u_exact, save_dir=save_directory)
    
    check_mass_conservation(u_pred, v_pred, dx, dy)
    
    plot_slices(u_pred, u_exact, x, y, x_slice=0.5, save_dir=save_directory)
    
    print("All visualizations generated successfully.")
