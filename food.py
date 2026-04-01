#!/usr/bin/env python3
"""
holagraph_v1_3d_poc.py

Holagraph v1.0 – 3D proof-of-concept:

- 2D emitter grid in (x, y)
- Field computed on a 2D plane at z = z0 (3D coordinates: x, y, z0)
- Intensity I(x, y, z0)
- 3D bead simulation: positions (x, y, z)
  * Forces from lateral intensity gradients (∂I/∂x, ∂I/∂y)
  * z motion is simple confinement / weak restoring
- Stable, clamped dynamics
- Plots:
  * Intensity map (x, y)
  * Bead distribution projected onto (x, y)
  * 3D scatter of bead cloud

NO hardware, NO serial, NO camera.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# GEOMETRY
# ============================================================

def emitter_grid(Nx, Ny, dx, dy):
    """
    Return 2D emitter positions as arrays (x_emit, y_emit) of shape (Nx*Ny,).
    """
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X.ravel(), Y.ravel()


def chamber_height():
    """
    Fixed chamber height above emitter plane.
    """
    return 1.5e-3  # 1.5 mm


# ============================================================
# PATTERNS
# ============================================================

def two_spot_pattern_2d(Nx, Ny, dx, dy, lambda_, spot1, spot2):
    """
    Two-spot pattern on a 2D emitter grid.

    spot1, spot2: (x, y) positions in meters where we want attractors.
    """
    k = 2 * np.pi / lambda_

    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    x1, y1 = spot1
    x2, y2 = spot2

    r1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
    r2 = np.sqrt((X - x2)**2 + (Y - y2)**2)

    phi1 = -k * r1
    phi2 = -k * r2

    phi = 0.5 * (phi1 + phi2)
    amp = np.ones_like(phi)

    return amp.ravel(), phi.ravel()


# ============================================================
# FIELD MODEL (2D PLANE AT z0)
# ============================================================

def field_2d_plane(x_grid, y_grid, x_emit, y_emit, amp, phi, lambda_, z0):
    """
    Compute complex field E(x, y, z0) from 2D emitter array.

    x_grid, y_grid: 2D meshgrid arrays (same shape)
    x_emit, y_emit: 1D arrays of emitter positions
    amp, phi: 1D arrays of emitter amplitudes and phases
    """
    k = 2 * np.pi / lambda_
    E = np.zeros_like(x_grid, dtype=complex)

    for i in range(len(x_emit)):
        dx = x_grid - x_emit[i]
        dy = y_grid - y_emit[i]
        r = np.sqrt(dx**2 + dy**2 + z0**2)
        G = np.exp(1j * k * r) / (r + 1e-9)
        E += amp[i] * np.exp(1j * phi[i]) * G

    return E


def intensity(E):
    return np.abs(E)**2


# ============================================================
# 3D BEAD SIMULATION
# ============================================================

def simulate_beads_3d(x0, y0, z0, I_grid, x_grid, y_grid,
                      dt, steps, alpha_xy, alpha_z, D_xy, D_z,
                      z_min, z_max):
    """
    3D bead simulation:

    - x, y forces from intensity gradients ∂I/∂x, ∂I/∂y
    - z has a weak restoring force toward mid-plane plus diffusion
    - positions clamped to domain
    """
    x = x0.copy()
    y = y0.copy()
    z = z0.copy()

    # Gradients on the 2D plane
    dI_dx, dI_dy = np.gradient(I_grid, x_grid[0, :], y_grid[:, 0])

    # Normalize gradients to avoid runaway
    max_g = np.max(np.sqrt(dI_dx**2 + dI_dy**2)) + 1e-12
    dI_dx /= max_g
    dI_dy /= max_g

    x_min, x_max = x_grid.min(), x_grid.max()
    y_min, y_max = y_grid.min(), y_grid.max()

    z_mid = 0.5 * (z_min + z_max)

    for _ in range(steps):
        # Interpolate gradients at bead positions
        dI_dx_beads = bilinear_interp(x, y, x_grid, y_grid, dI_dx)
        dI_dy_beads = bilinear_interp(x, y, x_grid, y_grid, dI_dy)

        Fx = 0.5 * alpha_xy * dI_dx_beads
        Fy = 0.5 * alpha_xy * dI_dy_beads

        # Weak restoring force in z toward mid-plane
        Fz = -alpha_z * (z - z_mid)

        # Update positions with drift + diffusion
        x += Fx * dt + np.sqrt(2 * D_xy * dt) * np.random.randn(*x.shape)
        y += Fy * dt + np.sqrt(2 * D_xy * dt) * np.random.randn(*y.shape)
        z += Fz * dt + np.sqrt(2 * D_z * dt) * np.random.randn(*z.shape)

        # Clamp to domain
        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        z = np.clip(z, z_min, z_max)

    return x, y, z


def bilinear_interp(xp, yp, x_grid, y_grid, F):
    """
    Bilinear interpolation of F(x, y) defined on regular grid (x_grid, y_grid)
    at points (xp, yp).

    x_grid, y_grid: 2D meshgrids
    F: same shape as x_grid, y_grid
    xp, yp: 1D arrays of query points
    """
    xs = x_grid[0, :]
    ys = y_grid[:, 0]

    Nx = len(xs)
    Ny = len(ys)

    ix = np.searchsorted(xs, xp) - 1
    iy = np.searchsorted(ys, yp) - 1

    ix = np.clip(ix, 0, Nx - 2)
    iy = np.clip(iy, 0, Ny - 2)

    x1 = xs[ix]
    x2 = xs[ix + 1]
    y1 = ys[iy]
    y2 = ys[iy + 1]

    tx = (xp - x1) / (x2 - x1 + 1e-12)
    ty = (yp - y1) / (y2 - y1 + 1e-12)

    f11 = F[iy, ix]
    f21 = F[iy, ix + 1]
    f12 = F[iy + 1, ix]
    f22 = F[iy + 1, ix + 1]

    f_interp = (
        f11 * (1 - tx) * (1 - ty) +
        f21 * tx * (1 - ty) +
        f12 * (1 - tx) * ty +
        f22 * tx * ty
    )

    return f_interp


# ============================================================
# MAIN 3D POC
# ============================================================

def run_3d_poc():
    # Emitter grid
    Nx, Ny = 16, 16
    dx, dy = 0.01, 0.01
    lambda_ = 850e-9
    z0 = chamber_height()

    x_emit, y_emit = emitter_grid(Nx, Ny, dx, dy)

    # Field sampling grid (plane at z0)
    x_grid_1d = np.linspace(0, (Nx - 1) * dx, 200)
    y_grid_1d = np.linspace(0, (Ny - 1) * dy, 200)
    Xg, Yg = np.meshgrid(x_grid_1d, y_grid_1d, indexing='xy')

    # Two spots in (x, y)
    spot1 = (4 * dx, 4 * dy)
    spot2 = (12 * dx, 12 * dy)

    amp, phi = two_spot_pattern_2d(Nx, Ny, dx, dy, lambda_, spot1, spot2)

    # Field + intensity on plane
    E = field_2d_plane(Xg, Yg, x_emit, y_emit, amp, phi, lambda_, z0)
    I = intensity(E)

    # 3D bead simulation
    n_beads = 1000
    x0 = np.random.uniform(Xg.min(), Xg.max(), size=n_beads)
    y0 = np.random.uniform(Yg.min(), Yg.max(), size=n_beads)
    z_min, z_max = 0.0, 3.0e-3
    z0_beads = np.random.uniform(z_min, z_max, size=n_beads)

    alpha_xy = 0.1
    alpha_z = 5.0
    D_xy = 1e-7
    D_z = 1e-8
    dt = 0.005
    steps = 800

    xf, yf, zf = simulate_beads_3d(
        x0, y0, z0_beads,
        I, Xg, Yg,
        dt, steps,
        alpha_xy, alpha_z,
        D_xy, D_z,
        z_min, z_max
    )

    # ============================================================
    # PLOTS
    # ============================================================

    plt.figure(figsize=(12, 10))

    # Intensity map
    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(
        I,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax1, label="Intensity")
    ax1.set_title("Holagraph v1.0 Intensity Map (Two-Spot, 2D Plane)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    # Bead projection on (x, y)
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist2d(xf, yf, bins=40, range=[[Xg.min(), Xg.max()], [Yg.min(), Yg.max()]],
               cmap='Blues')
    ax2.set_title("Bead Distribution Projection (x, y)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    plt.colorbar(ax2.collections[0], ax=ax2, label="Bead Count")

    # 3D scatter of beads
    ax3 = plt.subplot(2, 1, 2, projection='3d')
    ax3.scatter(xf, yf, zf, s=5, c=zf, cmap='viridis', alpha=0.7)
    ax3.set_title("3D Bead Cloud After Simulation")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_xlim(Xg.min(), Xg.max())
    ax3.set_ylim(Yg.min(), Yg.max())
    ax3.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.savefig("holagraph_v1_3d_poc.png")
    plt.show()

    print("3D proof-of-concept complete. Output saved to holagraph_v1_3d_poc.png")


if __name__ == "__main__":
    run_3d_poc()
