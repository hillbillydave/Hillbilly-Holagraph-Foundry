#!/usr/bin/env python3
"""
Holagraph v2.3.3 – Max-Stable, High-Power, Wide-Footprint Bowl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, gaussian_filter, label
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# TUNING PARAMETERS (ALL KNOBS HERE)
# =========================================================

# Emitter grid
EMITTER_N = 40
DX = DY = 0.01

# 3D grid resolution
GRID_RES = 200
Z_RES = 120
Z_MIN = 0.0
Z_MAX = 3.0e-3

# Wavelength
LAMBDA = 850e-9

# Amplitude boost (POWER CONTROL)
AMP_BOOST = 10.0     # 1.0 normal, 2.0 strong, 3.0 very strong

# Bowl width (SHAPE CONTROL)
SIGMA = 0.12        # 0.10–0.15 is ideal

# TIM exposure
EXPOSURE_TIME = 20.0
DT = 0.1

# TIM hardening rates
K_A = 1.2
K_B = 1.6
K_C = 2.0

# TIM thresholds
T_A = 0.40
T_B = 0.50
T_C = 0.80

# Morphology + smoothing
MORPH_KERNEL = 3
SMOOTH_SIGMA = 0.30
SMOOTH_THRESH = 0.05

# Power scaling
POWER_SCALE = 0.06


# =========================================================
# EMITTER GRID
# =========================================================

def emitter_grid(Nx, Ny, dx, dy):
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X.ravel(), Y.ravel()


# =========================================================
# FIELD PATTERN (WIDE BOWL)
# =========================================================

def deep_bowl_pattern(Nx, Ny, dx, dy, lambda_, focus, sigma):
    k = 2 * np.pi / lambda_
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    fx, fy = focus
    r = np.sqrt((X - fx)**2 + (Y - fy)**2)

    phi = -k * r
    amp = np.exp(-r**2 / (2 * sigma**2))

    return amp.ravel(), phi.ravel()


# =========================================================
# 3D FIELD VOLUME
# =========================================================

def field_3d_volume(x_grid, y_grid, z_vals, x_emit, y_emit, amp, phi, lambda_):
    k = 2 * np.pi / lambda_
    Nxg, Nyg = x_grid.shape
    Nz = len(z_vals)
    I_vol = np.zeros((Nz, Nyg, Nxg), dtype=float)

    for iz, z0 in enumerate(z_vals):
        E = np.zeros_like(x_grid, dtype=complex)
        for i in range(len(x_emit)):
            dx = x_grid - x_emit[i]
            dy = y_grid - y_emit[i]
            r = np.sqrt(dx**2 + dy**2 + z0**2)
            G = np.exp(1j * k * r) / (r + 1e-9)
            E += amp[i] * np.exp(1j * phi[i]) * G
        I_vol[iz] = np.abs(E)**2

    return I_vol


# =========================================================
# INTENSITY BANDS
# =========================================================

def segment_intensity_bands(I_vol, bands=(0.35, 0.60, 0.85)):
    I_max = I_vol.max()
    t_low, t_mid, t_high = [b * I_max for b in bands]

    low_mask = (I_vol >= t_low) & (I_vol < t_mid)
    mid_mask = (I_vol >= t_mid) & (I_vol < t_high)
    high_mask = I_vol >= t_high

    return low_mask, mid_mask, high_mask, I_max


# =========================================================
# VOXEL STATS
# =========================================================

def voxel_stats(mask, Xg, Yg, Zg):
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return None
    iz, iy, ix = idx
    xs = Xg[0, ix]
    ys = Yg[iy, 0]
    zs = Zg[iz]
    return {
        "centroid": (xs.mean(), ys.mean(), zs.mean()),
        "x_range": (xs.min(), xs.max()),
        "y_range": (ys.min(), ys.max()),
        "z_range": (zs.min(), zs.max()),
        "voxel_count": len(xs)
    }


# =========================================================
# STL EXPORT
# =========================================================

def write_voxel_stl(Xg, Yg, Zg, mask, filename):
    dx = Xg[0, 1] - Xg[0, 0]
    dy = Yg[1, 0] - Yg[0, 0]
    dz = Zg[1] - Zg[0]

    def cube_vertices(x, y, z):
        return np.array([
            [x, y, z],
            [x+dx, y, z],
            [x+dx, y+dy, z],
            [x, y+dy, z],
            [x, y, z+dz],
            [x+dx, y, z+dz],
            [x+dx, y+dy, z+dz],
            [x, y+dy, z+dz],
        ])

    def facet(f, n, v0, v1, v2):
        f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
        f.write("    outer loop\n")
        f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
        f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
        f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")

    Nz, Ny, Nx = mask.shape
    with open(filename, "w") as f:
        f.write(f"solid {filename}\n")

        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx):
                    if not mask[iz, iy, ix]:
                        continue

                    x = Xg[0, ix]
                    y = Yg[iy, 0]
                    z = Zg[iz]
                    V = cube_vertices(x, y, z)

                    faces = [
                        (0, 1, 2, 3),
                        (4, 5, 6, 7),
                        (0, 1, 5, 4),
                        (1, 2, 6, 5),
                        (2, 3, 7, 6),
                        (3, 0, 4, 7),
                    ]

                    for a, b, c, d in faces:
                        v0, v1, v2, v3 = V[a], V[b], V[c], V[d]
                        n = np.cross(v1 - v0, v2 - v0)
                        n /= (np.linalg.norm(n) + 1e-12)
                        facet(f, n, v0, v1, v2)
                        facet(f, n, v0, v2, v3)

        f.write(f"endsolid {filename}\n")


# =========================================================
# MAIN v2.3.3
# =========================================================

def run_v2_3_3():

    # Emitters
    Nx = Ny = EMITTER_N
    x_emit, y_emit = emitter_grid(Nx, Ny, DX, DY)

    # 3D grid
    x_grid_1d = np.linspace(0, (Nx - 1) * DX, GRID_RES)
    y_grid_1d = np.linspace(0, (Ny - 1) * DY, GRID_RES)
    Xg, Yg = np.meshgrid(x_grid_1d, y_grid_1d, indexing='xy')

    Zg = np.linspace(Z_MIN, Z_MAX, Z_RES)

    # Field pattern
    focus = (0.5 * (Nx - 1) * DX, 0.5 * (Ny - 1) * DY)
    amp, phi = deep_bowl_pattern(Nx, Ny, DX, DY, LAMBDA, focus, SIGMA)

    # POWER BOOST
    amp *= AMP_BOOST

    print("Computing 3D field volume...")
    I_vol = field_3d_volume(Xg, Yg, Zg, x_emit, y_emit, amp, phi, LAMBDA)

    # Power report
    avg_amp2 = np.mean(amp**2)
    slab_power = avg_amp2 * POWER_SCALE * len(amp)

    slice_power = I_vol.sum(axis=(1, 2))
    slice_power_norm = slice_power / slice_power.max()
    total_power_proxy = slice_power.sum()

    print("\n=== Holagraph v2.3.3 Power Report ===")
    print(f"Estimated slab power: {slab_power:.2f} W")
    print(f"Relative power (z): min={slice_power_norm.min():.3f}, max={slice_power_norm.max():.3f}")
    print(f"Total integrated intensity: {total_power_proxy:.3e}")
    print("=====================================")

    # Bands
    low_mask, mid_mask, high_mask, I_max = segment_intensity_bands(I_vol)

    print("\n=== Multi-Material Bands ===")
    print(f"I_max: {I_max:.3e}")
    print("A: 0.35–0.60   B: 0.60–0.85   C: 0.85–1.00")
    print("===================================")

    # TIM dose
    steps = int(EXPOSURE_TIME / DT)

    dose_A = np.zeros_like(I_vol)
    dose_B = np.zeros_like(I_vol)
    dose_C = np.zeros_like(I_vol)

    I_norm = I_vol / (I_max + 1e-12)

    print("\nAccumulating TIM dose...")
    for _ in range(steps):
        dose_A += K_A * I_norm * low_mask * DT
        dose_B += K_B * I_norm * mid_mask * DT
        dose_C += K_C * I_norm * high_mask * DT

    # TEMPORARY: bypass dose logic
    TIM_solid = I_vol > (0.6 * I_max)


    # Morphology
    print("\n=== v2.3.3 Morphology Refinement ===")

    structure = np.ones((MORPH_KERNEL, MORPH_KERNEL, MORPH_KERNEL))
    closed = binary_closing(TIM_solid, structure=structure)

    smooth = gaussian_filter(closed.astype(float), sigma=SMOOTH_SIGMA)
    refined = smooth > SMOOTH_THRESH

    labeled, num = label(refined)
    if num > 0:
        sizes = [(labeled == i).sum() for i in range(1, num+1)]
        largest = 1 + np.argmax(sizes)
        refined_solid = (labeled == largest)
    else:
        refined_solid = refined

    stats = voxel_stats(refined_solid, Xg, Yg, Zg)

    print("Refined TIM Solid:")
    if stats:
        print(stats)
    else:
        print("  No refined voxels found.")

    # STL
    write_voxel_stl(Xg, Yg, Zg, refined_solid, "deep_bowl_v2_3_3_TIMsolid_refined.stl")
    print("\nSTL export complete.")

    # Z-slice grid
    print("\nRendering full Z-slice grid...")

    cols = 8
    rows = int(np.ceil(Z_RES / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(Z_RES):
        ax = axes[i]
        ax.imshow(
            refined_solid[i].astype(float),
            extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
            origin='lower',
            cmap='Greens',
            aspect='equal'
        )
        ax.set_title(f"z = {Zg[i]*1e3:.2f} mm", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(Z_RES, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("holagraph_v2_3_3_TIM_slices.png")
    plt.show()

    print("\nHolagraph v2.3.3 run complete.")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    run_v2_3_3()
