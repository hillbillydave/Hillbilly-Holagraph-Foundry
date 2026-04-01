#!/usr/bin/env python3
"""
Holagraph v1.6 – Deep Bowl Solid Object

- Deeper, tighter bowl (narrower sigma)
- Stronger confinement in x, y, z
- Two masks:
    * Bowl footprint (outer basin)
    * Solid core (inner high-intensity region)
- 3D bead cloud in the deep bowl
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# GEOMETRY
# ============================================================

def emitter_grid(Nx, Ny, dx, dy):
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X.ravel(), Y.ravel()


def chamber_height():
    return 1.5e-3  # 1.5 mm


# ============================================================
# DEEP BOWL PATTERN
# ============================================================

def deep_bowl_pattern(Nx, Ny, dx, dy, lambda_, focus):
    k = 2 * np.pi / lambda_
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    fx, fy = focus
    r = np.sqrt((X - fx)**2 + (Y - fy)**2)

    # Phase for convergence
    phi = -k * r

    # Narrower Gaussian → deeper, tighter bowl
    sigma = 0.02  # v1.5 used ~0.04
    amp = np.exp(-r**2 / (2 * sigma**2))

    return amp.ravel(), phi.ravel()


# ============================================================
# FIELD MODEL (2D PLANE)
# ============================================================

def field_2d_plane(x_grid, y_grid, x_emit, y_emit, amp, phi, lambda_, z0):
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
# BILINEAR INTERPOLATION
# ============================================================

def bilinear_interp(xp, yp, x_grid, y_grid, F):
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
# 3D BEAD SIMULATION (DEEP BOWL FIELD)
# ============================================================

def simulate_beads_3d(
    x0, y0, z0_beads,
    x_emit, y_emit,
    x_grid, y_grid,
    lambda_, z_plane,
    dt, steps,
    alpha_xy, alpha_z,
    D_xy, D_z,
    z_min, z_max,
    amp, phi,
    power_scale_watts=0.06
):
    x = x0.copy()
    y = y0.copy()
    z = z0_beads.copy()

    x_min, x_max = x_grid.min(), x_grid.max()
    y_min, y_max = y_grid.min(), y_grid.max()
    z_mid = 0.5 * (z_min + z_max)

    power_time = []

    for step in range(steps):
        E = field_2d_plane(x_grid, y_grid, x_emit, y_emit, amp, phi, lambda_, z_plane)
        I = intensity(E)

        dI_dx, dI_dy = np.gradient(I, x_grid[0, :], y_grid[:, 0])
        max_g = np.max(np.sqrt(dI_dx**2 + dI_dy**2)) + 1e-12
        dI_dx /= max_g
        dI_dy /= max_g

        dI_dx_beads = bilinear_interp(x, y, x_grid, y_grid, dI_dx)
        dI_dy_beads = bilinear_interp(x, y, x_grid, y_grid, dI_dy)

        Fx = 0.8 * alpha_xy * dI_dx_beads   # stronger pull than v1.5
        Fy = 0.8 * alpha_xy * dI_dy_beads
        Fz = -alpha_z * (z - z_mid)

        x += Fx * dt + np.sqrt(2 * D_xy * dt) * np.random.randn(*x.shape)
        y += Fy * dt + np.sqrt(2 * D_xy * dt) * np.random.randn(*y.shape)
        z += Fz * dt + np.sqrt(2 * D_z * dt) * np.random.randn(*z.shape)

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        z = np.clip(z, z_min, z_max)

        avg_amp2 = np.mean(amp**2)
        total_power = avg_amp2 * power_scale_watts * len(amp)
        power_time.append(total_power)

    power_time = np.array(power_time)

    return x, y, z, I, power_time


# ============================================================
# OBJECT IDENTIFICATION (BOWL + SOLID CORE)
# ============================================================

def identify_bowl_and_core(I, Xg, Yg, bowl_frac=0.30, core_frac=0.60):
    I_max = I.max()
    bowl_thresh = bowl_frac * I_max
    core_thresh = core_frac * I_max

    bowl_mask = I >= bowl_thresh
    core_mask = I >= core_thresh

    ys, xs = np.where(bowl_mask)
    if len(xs) == 0:
        return None, None, bowl_mask, core_mask

    x_coords = Xg[0, xs]
    y_coords = Yg[ys, 0]

    x_centroid = x_coords.mean()
    y_centroid = y_coords.mean()

    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()

    bowl_obj = {
        "x_centroid": x_centroid,
        "y_centroid": y_centroid,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "area_pixels": int(bowl_mask.sum())
    }

    ys_c, xs_c = np.where(core_mask)
    if len(xs_c) == 0:
        core_obj = None
    else:
        x_core = Xg[0, xs_c]
        y_core = Yg[ys_c, 0]
        core_obj = {
            "x_range": (x_core.min(), x_core.max()),
            "y_range": (y_core.min(), y_core.max()),
            "area_pixels": int(core_mask.sum())
        }

    return bowl_obj, core_obj, bowl_mask, core_mask


# ============================================================
# MAIN v1.6 RUN – DEEP BOWL
# ============================================================

def run_v1_6():
    Nx, Ny = 16, 16
    dx, dy = 0.01, 0.01
    lambda_ = 850e-9
    z_plane = chamber_height()

    x_emit, y_emit = emitter_grid(Nx, Ny, dx, dy)

    x_grid_1d = np.linspace(0, (Nx - 1) * dx, 220)
    y_grid_1d = np.linspace(0, (Ny - 1) * dy, 220)
    Xg, Yg = np.meshgrid(x_grid_1d, y_grid_1d, indexing='xy')

    focus = (0.5 * (Nx - 1) * dx, 0.5 * (Ny - 1) * dy)
    amp, phi = deep_bowl_pattern(Nx, Ny, dx, dy, lambda_, focus)

    n_beads = 2000
    x0 = np.random.uniform(Xg.min(), Xg.max(), size=n_beads)
    y0 = np.random.uniform(Yg.min(), Yg.max(), size=n_beads)
    z_min, z_max = 0.0, 3.0e-3
    z0_beads = np.random.uniform(z_min, z_max, size=n_beads)

    alpha_xy = 0.3   # stronger than v1.5
    alpha_z = 10.0   # tighter vertical confinement
    D_xy = 5e-8      # slightly less diffusion
    D_z = 5e-9
    dt = 0.004
    steps = 700

    xf, yf, zf, I_final, power_time = simulate_beads_3d(
        x0, y0, z0_beads,
        x_emit, y_emit,
        Xg, Yg,
        lambda_, z_plane,
        dt, steps,
        alpha_xy, alpha_z,
        D_xy, D_z,
        z_min, z_max,
        amp, phi,
        power_scale_watts=0.06
    )

    bowl_obj, core_obj, bowl_mask, core_mask = identify_bowl_and_core(
        I_final, Xg, Yg, bowl_frac=0.30, core_frac=0.60
    )

    print("\n=== Holagraph v1.6 Deep Bowl Object Report ===")
    if bowl_obj is None:
        print("No bowl object detected at chosen thresholds.")
    else:
        print(f"Bowl centroid: ({bowl_obj['x_centroid']:.4f} m, {bowl_obj['y_centroid']:.4f} m)")
        print(f"Bowl X range:  {bowl_obj['x_range'][0]:.4f} m to {bowl_obj['x_range'][1]:.4f} m")
        print(f"Bowl Y range:  {bowl_obj['y_range'][0]:.4f} m to {bowl_obj['y_range'][1]:.4f} m")
        print(f"Bowl area (pixels): {bowl_obj['area_pixels']}")
    if core_obj is not None:
        print("\nSolid core region:")
        print(f"Core X range:  {core_obj['x_range'][0]:.4f} m to {core_obj['x_range'][1]:.4f} m")
        print(f"Core Y range:  {core_obj['y_range'][0]:.4f} m to {core_obj['y_range'][1]:.4f} m")
        print(f"Core area (pixels): {core_obj['area_pixels']}")
    print("==============================================\n")

    print(f"Average power: {power_time.mean():.2f} W, peak: {power_time.max():.2f} W")

    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(
        I_final,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax1, label="Intensity")
    ax1.set_title("Holagraph v1.6 Deep Bowl Intensity Map")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(
        bowl_mask,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='Greens',
        aspect='equal'
    )
    ax2.contour(
        core_mask.astype(float),
        levels=[0.5],
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        colors='red',
        linewidths=1.0
    )
    ax2.set_title("Bowl Mask (Green) + Solid Core (Red Contour)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    ax3 = plt.subplot(2, 2, 3)
    h = ax3.hist2d(
        xf, yf,
        bins=45,
        range=[[Xg.min(), Xg.max()], [Yg.min(), Yg.max()]],
        cmap='Blues'
    )
    plt.colorbar(h[3], ax=ax3, label="Bead Count")
    ax3.set_title("Bead Distribution Projection (x, y)")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = plt.subplot(2, 2, 4, projection='3d')
    sc = ax4.scatter(xf, yf, zf, s=4, c=zf, cmap='viridis', alpha=0.7)
    ax4.set_title("3D Bead Cloud in Deep Bowl Field")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_zlabel("z (m)")
    ax4.set_xlim(Xg.min(), Xg.max())
    ax4.set_ylim(Yg.min(), Yg.max())
    ax4.set_zlim(z_min, z_max)
    plt.colorbar(sc, ax=ax4, label="z (m)")

    plt.tight_layout()
    plt.savefig("holagraph_v1_6_deep_bowl.png")
    plt.show()

    print("Holagraph v1.6 run complete. Output saved to holagraph_v1_6_deep_bowl.png")


if __name__ == "__main__":
    run_v1_6()
