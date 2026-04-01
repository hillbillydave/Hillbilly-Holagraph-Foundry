#!/usr/bin/env python3
"""
Holagraph v2.1 – Multi-Material Field Bands

- 3D intensity volume
- Power profile (z)
- Multi-band segmentation of intensity
- Separate voxel STL per band (A/B/C)
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
    """
    bands: (low_thresh, mid_thresh, high_thresh) as fractions of I_max
    Returns three boolean masks: low, mid, high
    """
    I_max = I_vol.max()
    t_low, t_mid, t_high = [b * I_max for b in bands]

    # low: between t_low and t_mid
    low_mask = (I_vol >= t_low) & (I_vol < t_mid)
    # mid: between t_mid and t_high
    mid_mask = (I_vol >= t_mid) & (I_vol < t_high)
    # high: >= t_high
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


def run_v2_1():
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

    print("\n=== Holagraph v2.1 Power Report ===")
    print(f"Estimated slab power (emitters): {slab_power:.2f} W")
    print(f"Relative power (z-slices): min={slice_power_norm.min():.3f}, "
          f"max={slice_power_norm.max():.3f}")
    print(f"Total integrated intensity proxy: {total_power_proxy:.3e}")
    print("===================================")

    # Multi-material segmentation
    low_mask, mid_mask, high_mask, I_max = segment_intensity_bands(
        I_vol, bands=(0.35, 0.60, 0.85)
    )

    low_stats = voxel_stats(low_mask, Xg, Yg, Zg)
    mid_stats = voxel_stats(mid_mask, Xg, Yg, Zg)
    high_stats = voxel_stats(high_mask, Xg, Yg, Zg)

    print("\n=== Holagraph v2.1 Multi-Material Bands ===")
    print(f"I_max: {I_max:.3e}")
    print("Band thresholds (fractions of I_max):")
    print("  Material A (low):   0.35–0.60")
    print("  Material B (mid):   0.60–0.85")
    print("  Material C (high):  0.85–1.00\n")

    def print_band(name, stats):
        if stats is None:
            print(f"{name}: no voxels")
        else:
            cx, cy, cz = stats["centroid"]
            print(f"{name}:")
            print(f"  Centroid: ({cx:.4f} m, {cy:.4f} m, {cz:.6f} m)")
            print(f"  X range:  {stats['x_range'][0]:.4f} m to {stats['x_range'][1]:.4f} m")
            print(f"  Y range:  {stats['y_range'][0]:.4f} m to {stats['y_range'][1]:.4f} m")
            print(f"  Z range:  {stats['z_range'][0]:.6f} m to {stats['z_range'][1]:.6f} m")
            print(f"  Voxel count: {stats['voxel_count']}")

    print_band("Material A (low)", low_stats)
    print()
    print_band("Material B (mid)", mid_stats)
    print()
    print_band("Material C (high)", high_stats)
    print("============================================")

    # STL exports per band
    write_voxel_stl(Xg, Yg, Zg, low_mask,  "deep_bowl_v2_1_matA_low.stl")
    write_voxel_stl(Xg, Yg, Zg, mid_mask,  "deep_bowl_v2_1_matB_mid.stl")
    write_voxel_stl(Xg, Yg, Zg, high_mask, "deep_bowl_v2_1_matC_high.stl")
    print("\nSTL exports complete:")
    print("  deep_bowl_v2_1_matA_low.stl")
    print("  deep_bowl_v2_1_matB_mid.stl")
    print("  deep_bowl_v2_1_matC_high.stl")

    # Quick plots
    mid = Nz // 2
    I_mid = I_vol[mid]
    low_mid = low_mask[mid]
    mid_mid = mid_mask[mid]
    high_mid = high_mask[mid]

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
    ax1.set_title("v2.1 Central z-Slice Intensity")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(Zg * 1e3, slice_power_norm, '-o')
    ax2.set_title("Relative Power vs z")
    ax2.set_xlabel("z (mm)")
    ax2.set_ylabel("Relative power")

    ax3 = fig.add_subplot(2, 2, 3)
    # overlay band masks in central slice
    ax3.imshow(
        low_mid.astype(float),
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='Greens',
        alpha=0.5,
        aspect='equal'
    )
    ax3.imshow(
        mid_mid.astype(float),
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='Blues',
        alpha=0.5,
        aspect='equal'
    )
    ax3.imshow(
        high_mid.astype(float),
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='Reds',
        alpha=0.5,
        aspect='equal'
    )
    ax3.set_title("Central Slice – A/B/C Bands")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for mask_band, cmap, label in [
        (low_mask, "Greens", "A low"),
        (mid_mask, "Blues", "B mid"),
        (high_mask, "Reds", "C high"),
    ]:
        iz, iy, ix = np.where(mask_band)
        if len(iz) == 0:
            continue
        xs = Xg[0, ix]
        ys = Yg[iy, 0]
        zs = Zg[iz]
        ax4.scatter(xs, ys, zs, s=4, alpha=0.7, cmap=cmap, c=zs, label=label)
    ax4.set_title("3D Multi-Material Voxels")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_zlabel("z (m)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("holagraph_v2_1_multimaterial.png")
    plt.show()

    print("\nHolagraph v2.1 run complete. Outputs saved.")


if __name__ == "__main__":
    run_v2_1()
