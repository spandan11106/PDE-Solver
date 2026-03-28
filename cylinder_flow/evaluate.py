import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from network import PINN

os.makedirs('results', exist_ok=True)
device = torch.device('cpu')

model = PINN().to(device)
if os.path.exists('pinn_cylinder_refined.pth'):
    model_path = 'pinn_cylinder_refined.pth'
elif os.path.exists('pinn_cylinder.pth'):
    model_path = 'pinn_cylinder.pth'
else:
    raise FileNotFoundError("Weights not found.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

t_val = 0.5 
x = np.linspace(-1.0, 5.0, 200)
y = np.linspace(-2.0, 2.0, 150)
X, Y = np.meshgrid(x, y)

t_grid = np.full((X.size, 1), t_val)
grid_points = np.hstack([t_grid, X.reshape(-1, 1), Y.reshape(-1, 1)])
grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

with torch.no_grad():
    preds = model(grid_tensor)
    u_pred = preds[:, 0].cpu().numpy().reshape(X.shape)
    v_pred = preds[:, 1].cpu().numpy().reshape(X.shape)
    p_pred = preds[:, 2].cpu().numpy().reshape(X.shape)

vel_mag = np.sqrt(u_pred**2 + v_pred**2)

cyl_r = 0.5
mask = (X**2 + Y**2) <= cyl_r**2
u_pred[mask] = np.nan
v_pred[mask] = np.nan
p_pred[mask] = np.nan
vel_mag[mask] = np.nan

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Cylinder Wake Predictions at t = {t_val}s", fontsize=16, fontweight='bold')

def plot_field(ax, field, title, vmin, vmax, cmap='jet'):
    im = ax.contourf(X, Y, field, levels=np.linspace(vmin, vmax, 50), cmap=cmap, extend='both')
    ax.set_title(title, fontsize=14)
    cylinder_patch = plt.Circle((0, 0), cyl_r, color='dimgray', zorder=10)
    ax.add_patch(cylinder_patch)
    fig.colorbar(im, ax=ax, ticks=np.linspace(vmin, vmax, 7))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

plot_field(axes[0, 0], u_pred, 'X-Velocity ($u$)', vmin=-0.5, vmax=1.5)
plot_field(axes[0, 1], v_pred, 'Y-Velocity ($v$)', vmin=-0.8, vmax=0.8)
plot_field(axes[1, 0], p_pred, 'Pressure ($p$)', vmin=-0.5, vmax=0.5)
plot_field(axes[1, 1], vel_mag, 'Velocity Magnitude ($||V||$)', vmin=0.0, vmax=1.5)

plt.tight_layout()
save_path = os.path.join('results', f'cylinder_eval_t{str(t_val).replace(".", "_")}.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to '{save_path}' with fixed scales!")