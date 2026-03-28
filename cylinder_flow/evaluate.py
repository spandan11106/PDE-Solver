import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from network import PINN

# 1. Ensure results directory exists inside cylinder_flow
os.makedirs('results', exist_ok=True)

# Force CPU execution per your setup
device = torch.device('cpu')
print(f"Evaluating on device: {device}")

# 2. Load the model
model = PINN().to(device)

# Intelligently load refined weights if they exist, otherwise fallback to base weights
if os.path.exists('pinn_cylinder_refined.pth'):
    model_path = 'pinn_cylinder_refined.pth'
elif os.path.exists('pinn_cylinder.pth'):
    model_path = 'pinn_cylinder.pth'
else:
    raise FileNotFoundError("Could not find 'pinn_cylinder.pth' or 'pinn_cylinder_refined.pth'. Please train the model first.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded weights from {model_path}")

# 3. Create a dense grid for plotting at a specific time snapshot
t_val = 0.5  
x = np.linspace(-1.0, 5.0, 200)
y = np.linspace(-2.0, 2.0, 150)
X, Y = np.meshgrid(x, y)

# Flatten grid and attach the time dimension for the network (t, x, y)
t_grid = np.full((X.size, 1), t_val)
grid_points = np.hstack([t_grid, X.reshape(-1, 1), Y.reshape(-1, 1)])
grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

# 4. Get Network Predictions
print("Computing predictions across the grid...")
with torch.no_grad():
    preds = model(grid_tensor)
    u_pred = preds[:, 0].cpu().numpy().reshape(X.shape)
    v_pred = preds[:, 1].cpu().numpy().reshape(X.shape)
    p_pred = preds[:, 2].cpu().numpy().reshape(X.shape)

# Calculate velocity magnitude: ||V|| = sqrt(u^2 + v^2)
vel_mag = np.sqrt(u_pred**2 + v_pred**2)

# 5. Mask out the interior of the cylinder for realistic visualization
cyl_r = 0.5
mask = (X**2 + Y**2) <= cyl_r**2
u_pred[mask] = np.nan
v_pred[mask] = np.nan
p_pred[mask] = np.nan
vel_mag[mask] = np.nan

# 6. Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Cylinder Wake PINN Predictions at t = {t_val}s", fontsize=16, fontweight='bold')

# Helper function to plot fields and draw the cylinder
def plot_field(ax, field, title, cmap='jet'):
    # Plot the contour
    im = ax.contourf(X, Y, field, 50, cmap=cmap)
    ax.set_title(title, fontsize=14)
    # Draw the physical cylinder
    cylinder_patch = plt.Circle((0, 0), cyl_r, color='dimgray', zorder=10)
    ax.add_patch(cylinder_patch)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

# Render the 4 subplots
plot_field(axes[0, 0], u_pred, 'X-Velocity ($u$)')
plot_field(axes[0, 1], v_pred, 'Y-Velocity ($v$)')
plot_field(axes[1, 0], p_pred, 'Pressure ($p$)')
plot_field(axes[1, 1], vel_mag, 'Velocity Magnitude ($||V||$)')

plt.tight_layout()

# Save the plot
save_path = os.path.join('results', f'cylinder_eval_t{int(t_val)}.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to '{save_path}'!")