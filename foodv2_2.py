#!/usr/bin/env python3
"""
Holagraph v2.2 – TIM Layer (Multi-Material Hardening)

- 3D intensity volume
- Power profile (z)
- Multi-band segmentation (A/B/C)
- TIM dose accumulation with band-dependent hardening
- Final TIM-hardened solid STL
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def emitter_grid(Nx, Ny, dx, dy):
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X.ravel(), Y.ravel()


def deep_bowl_pattern(Nx, Ny, dx, dy, lambda_, focus):
    k = 2 * np.pi / lambda_
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    fx, fy = focus
    r = np.sqrt((X - fx)**2 + (Y - fy)**2)
    phi = -k * r
    sigma = 0.02
    amp = np.exp(-r**2 / (2 * sigma**2))
    return amp.ravel(), phi.ravel()


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


def segment_intensity_bands(I_vol, bands=(0.35, 0.60, 0.85)):
    I_max = I_vol.max()
    t_low, t_mid, t_high = [b * I_max for b in bands]
    low_mask = (I_vol >= t_low) & (I_vol < t_mid)
    mid_mask = (I_vol >= t_mid) & (I_vol < t_high)
    high_mask = I_vol >= t_high
    return low_mask, mid_mask, high_mask, I_max


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


def write_voxel_stl(Xg, Yg, Zg, mask, filename):
    dx = Xg[0, 1] - Xg[0, 0]
    dy = Yg[1, 0] - Yg[0, 0]
    dz = Zg[1] - Zg[0]

    def cube_vertices(x, y, z):
        return np.array([
            [x,       y,       z      ],
            [x+dx,    y,       z      ],
            [x+dx,    y+dy,    z      ],
            [x,       y+dy,    z      ],
            [x,       y,       z+dz   ],
            [x+dx,    y,       z+dz   ],
            [x+dx,    y+dy,    z+dz   ],
            [x,       y+dy,    z+dz   ],
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


def run_v2_2():
    Nx, Ny = 16, 16
    dx, dy = 0.01, 0.01
    lambda_ = 850e-9

    x_emit, y_emit = emitter_grid(Nx, Ny, dx, dy)

    x_grid_1d = np.linspace(0, (Nx - 1) * dx, 80)
    y_grid_1d = np.linspace(0, (Ny - 1) * dy, 80)
    Xg, Yg = np.meshgrid(x_grid_1d, y_grid_1d, indexing='xy')

    z_min, z_max = 0.0, 3.0e-3
    Nz = 40
    Zg = np.linspace(z_min, z_max, Nz)

    focus = (0.5 * (Nx - 1) * dx, 0.5 * (Ny - 1) * dy)
    amp, phi = deep_bowl_pattern(Nx, Ny, dx, dy, lambda_, focus)

    print("Computing 3D field volume...")
    I_vol = field_3d_volume(Xg, Yg, Zg, x_emit, y_emit, amp, phi, lambda_)

    # Power estimates
    power_scale = 0.06
    avg_amp2 = np.mean(amp**2)
    slab_power = avg_amp2 * power_scale * len(amp)

    slice_power = I_vol.sum(axis=(1, 2))
    slice_power_norm = slice_power / slice_power.max()
    total_power_proxy = slice_power.sum()

    print("\n=== Holagraph v2.2 Power Report ===")
    print(f"Estimated slab power (emitters): {slab_power:.2f} W")
    print(f"Relative power (z-slices): min={slice_power_norm.min():.3f}, "
          f"max={slice_power_norm.max():.3f}")
    print(f"Total integrated intensity proxy: {total_power_proxy:.3e}")
    print("===================================")

    # Multi-material segmentation
    low_mask, mid_mask, high_mask, I_max = segment_intensity_bands(
        I_vol, bands=(0.35, 0.60, 0.85)
    )

    print("\n=== Holagraph v2.2 Multi-Material Bands ===")
    print(f"I_max: {I_max:.3e}")
    print("Band thresholds (fractions of I_max):")
    print("  Material A (low):   0.35–0.60")
    print("  Material B (mid):   0.60–0.85")
    print("  Material C (high):  0.85–1.00")
    print("===================================")

    # --- TIM MODEL (multi-material hardening) ---

    # Exposure parameters (arbitrary units, but consistent)
    exposure_time = 1.0  # seconds
    dt = 0.1
    steps = int(exposure_time / dt)

    # Hardening rates per band (A/B/C)
    k_A = 0.8   # soft, slower
    k_B = 1.0   # medium
    k_C = 1.3   # firm, faster

    # Hardening thresholds per band (dose units)
    T_A = 0.4
    T_B = 0.6
    T_C = 0.8

    # Initialize dose fields
    dose_A = np.zeros_like(I_vol, dtype=float)
    dose_B = np.zeros_like(I_vol, dtype=float)
    dose_C = np.zeros_like(I_vol, dtype=float)

    # Normalize intensity to [0,1] for dose model
    I_norm = I_vol / (I_max + 1e-12)

    print("\nAccumulating TIM dose over time...")
    for _ in range(steps):
        dose_A += k_A * I_norm * low_mask * dt
        dose_B += k_B * I_norm * mid_mask * dt
        dose_C += k_C * I_norm * high_mask * dt

    # Hardened regions per band
    hard_A = dose_A >= T_A
    hard_B = dose_B >= T_B
    hard_C = dose_C >= T_C

    # Final TIM solid: union of all hardened bands
    TIM_solid = hard_A | hard_B | hard_C

    # Stats
    def band_stats(name, mask_band):
        stats = voxel_stats(mask_band, Xg, Yg, Zg)
        if stats is None:
            print(f"{name}: no hardened voxels")
        else:
            cx, cy, cz = stats["centroid"]
            print(f"{name}:")
            print(f"  Centroid: ({cx:.4f} m, {cy:.4f} m, {cz:.6f} m)")
            print(f"  X range:  {stats['x_range'][0]:.4f} m to {stats['x_range'][1]:.4f} m")
            print(f"  Y range:  {stats['y_range'][0]:.4f} m to {stats['y_range'][1]:.4f} m")
            print(f"  Z range:  {stats['z_range'][0]:.6f} m to {stats['z_range'][1]:.6f} m")
            print(f"  Voxel count: {stats['voxel_count']}")

    print("\n=== TIM Hardening Results (per band) ===")
    band_stats("Material A (soft) – hardened", hard_A)
    print()
    band_stats("Material B (medium) – hardened", hard_B)
    print()
    band_stats("Material C (firm) – hardened", hard_C)
    print("========================================")

    TIM_stats = voxel_stats(TIM_solid, Xg, Yg, Zg)
    print("\n=== Final TIM Solid (union A/B/C) ===")
    if TIM_stats is None:
        print("No hardened TIM solid at chosen parameters.")
    else:
        cx, cy, cz = TIM_stats["centroid"]
        print(f"Centroid: ({cx:.4f} m, {cy:.4f} m, {cz:.6f} m)")
        print(f"X range:  {TIM_stats['x_range'][0]:.4f} m to {TIM_stats['x_range'][1]:.4f} m")
        print(f"Y range:  {TIM_stats['y_range'][0]:.4f} m to {TIM_stats['y_range'][1]:.4f} m")
        print(f"Z range:  {TIM_stats['z_range'][0]:.6f} m to {TIM_stats['z_range'][1]:.6f} m")
        print(f"Voxel count: {TIM_stats['voxel_count']}")
    print("======================================")

    # STL export of final TIM solid
    write_voxel_stl(Xg, Yg, Zg, TIM_solid, "deep_bowl_v2_2_TIMsolid.stl")
    print("\nSTL export complete: deep_bowl_v2_2_TIMsolid.stl")

    # Quick plots
    mid = Nz // 2
    I_mid = I_vol[mid]
    TIM_mid = TIM_solid[mid]

    fig = plt.figure(figsize=(13, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(
        I_mid,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax1, label="Intensity")
    ax1.set_title("v2.2 Central z-Slice Intensity")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(Zg * 1e3, slice_power_norm, '-o')
    ax2.set_title("Relative Power vs z")
    ax2.set_xlabel("z (mm)")
    ax2.set_ylabel("Relative power")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(
        TIM_mid.astype(float),
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='Greens',
        aspect='equal'
    )
    ax3.set_title("Central Slice – Hardened TIM")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    iz, iy, ix = np.where(TIM_solid)
    if len(iz) > 0:
        xs = Xg[0, ix]
        ys = Yg[iy, 0]
        zs = Zg[iz]
        sc = ax4.scatter(xs, ys, zs, s=4, c=zs, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=ax4, label="z (m)")
    ax4.set_title("3D TIM-Hardened Solid")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_zlabel("z (m)")

    plt.tight_layout()
    plt.savefig("holagraph_v2_2_TIMsolid.png")
    plt.show()

    print("\nHolagraph v2.2 run complete. Outputs saved.")


if __name__ == "__main__":
    run_v2_2()
