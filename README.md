# PDE-Solver: PINN for Kovasznay Flow

A Physics-Informed Neural Network (PINN) implementation for solving the steady 2D incompressible Navier-Stokes equations on the Kovasznay benchmark flow.

The model predicts:
- `u(x, y)`: x-velocity
- `v(x, y)`: y-velocity
- `p(x, y)`: pressure

and is trained with a combination of:
- boundary-condition supervision from the analytical Kovasznay solution
- PDE residual minimization at interior collocation points

## Problem Setup

This project solves the steady incompressible Navier-Stokes system:

- Continuity: `u_x + v_y = 0`
- X-momentum: `u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy) = 0`
- Y-momentum: `u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy) = 0`

with Reynolds number:
- `Re = 20`

Domain:
- `x in [-0.5, 1.0]`
- `y in [-0.5, 0.5]`

## Repository Structure

```
PDE-Solver/
├── LICENSE
├── README.md
├── requirements.txt
├── pinn_kovasznay.pth
├── results/
│   ├── error_distribution.png
│   ├── kovasznay_results.png
│   ├── kovasznay_results_lbfgs.png
│   ├── physics_residuals.png
│   └── velocity_profile_x_0_5049999999999999.png
└── src/
    ├── dataset.py
    ├── evaluate.py
    ├── kovasznay.py
    ├── loss.py
    ├── network.py
    ├── train.py
    └── visualize.py
```

## Installation

1. Clone the repository and move into it.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies:
- `torch`
- `numpy`
- `matplotlib`

## Quick Start

Run all commands from the project root.

### 1. Train the PINN

```bash
python src/train.py
```

What happens during training:
- Stage 1: Adam optimizer with `ReduceLROnPlateau`
- Stage 2: L-BFGS fine-tuning
- Adaptive loss balancing via trainable weights `w_bnd` and `w_phys`
- Model is saved as `pinn_kovasznay.pth`

### 2. Evaluate Against Analytical Solution

```bash
python src/evaluate.py
```

This script:
- loads `pinn_kovasznay.pth`
- computes predictions on a dense `101 x 101` grid
- computes L2 relative errors for `u`, `v`, and `p`
- saves comparison plots to `kovasznay_results.png`

### 3. Generate Additional Diagnostics

```bash
python src/visualize.py
```

This script generates diagnostics in `results/`:
- `physics_residuals.png` (continuity residual map)
- `error_distribution.png` (log-scale error histogram)
- `velocity_profile_x_*.png` (slice comparison)
- printed net mass flux check

## Implementation Notes

### PINN Architecture

Defined in `src/network.py`:
- Input layer: 2 -> 50
- Hidden layers: 4 additional fully-connected layers of size 50
- Activation: `tanh`
- Output layer: 50 -> 3 (`u, v, p`)

### Data Sampling

Defined in `src/dataset.py`:
- Interior collocation points: 5000 random samples
- Boundary points: 800 random samples total (200 per side)
- Boundary targets come from analytical Kovasznay solution

### Loss Function

Defined in `src/loss.py`:
- `mse_bnd`: boundary-condition MSE for `u, v, p`
- `mse_physics`: PDE residual MSE for continuity and momentum equations
- Total training objective in `src/train.py` uses adaptive weighting:
  - `0.5 * exp(-w_bnd) * mse_bnd + 0.5 * w_bnd`
  - `0.5 * exp(-w_phys) * mse_phys + 0.5 * w_phys`

## Typical Workflow

```bash
python src/train.py
python src/evaluate.py
python src/visualize.py
```

Then inspect:
- `kovasznay_results.png`
- files inside `results/`

## Troubleshooting

- If CUDA is unavailable, training/evaluation automatically falls back to CPU.
- If `src/evaluate.py` cannot find weights, ensure `pinn_kovasznay.pth` exists in the project root.
- If you retrain, old result plots may be overwritten.

## License

This project is distributed under the terms in `LICENSE`.
