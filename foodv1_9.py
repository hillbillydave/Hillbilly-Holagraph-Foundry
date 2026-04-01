#!/usr/bin/env python3
"""
Holagraph v1.9 – Full 3D Voxel Solid

- Extends deep bowl into full 3D intensity volume
- Thresholds into a true volumetric voxel object
- Exports voxel solid as STL
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ================== BASIC GEOMETRY ==================

def emitter_grid(Nx, Ny, dx, dy):
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X.ravel(), Y.ravel()


def chamber_height():
    return 1.5e-3  # 1.5 mm


# ================== DEEP BOWL PATTERN ==================

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


# ================== FIELD MODEL ==================

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


# ================== VOXEL OBJECT ==================

def build_voxel_object(I_vol, Xg, Yg, Zg, frac=0.35):
    I_max = I_vol.max()
    thresh = frac * I_max
    mask = I_vol >= thresh

    idx = np.where(mask)
    if len(idx[0]) == 0:
        return None, mask

    iz, iy, ix = idx
    xs = Xg[0, ix]
    ys = Yg[iy, 0]
    zs = Zg[iz]

    centroid = (xs.mean(), ys.mean(), zs.mean())
    x_range = (xs.min(), xs.max())
    y_range = (ys.min(), ys.max())
    z_range = (zs.min(), zs.max())
    voxel_count = len(xs)

    obj = {
        "centroid": centroid,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "voxel_count": voxel_count
    }
    return obj, mask


# ================== STL EXPORT (VOXEL SOLID) ==================

def write_voxel_stl(Xg, Yg, Zg, mask, filename="deep_bowl_v1_9_voxel_solid.stl"):
    """
    Simple voxel STL: each active voxel becomes a small cube.
    Coarse but truly volumetric.
    """
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

    with open(filename, "w") as f:
        f.write("solid deep_bowl_v1_9_voxel_solid\n")

        Nz, Ny, Nx = mask.shape
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
                        (0, 1, 2, 3),  # bottom
                        (4, 5, 6, 7),  # top
                        (0, 1, 5, 4),  # side
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

        f.write("endsolid deep_bowl_v1_9_voxel_solid\n")


# ================== MAIN v1.9 ==================

def run_v1_9():
    Nx, Ny = 16, 16
    dx, dy = 0.01, 0.01
    lambda_ = 850e-9

    x_emit, y_emit = emitter_grid(Nx, Ny, dx, dy)

    x_grid_1d = np.linspace(0, (Nx - 1) * dx, 80)
    y_grid_1d = np.linspace(0, (Ny - 1) * dy, 80)
    Xg, Yg = np.meshgrid(x_grid_1d, y_grid_1d, indexing='xy')

    # 3D volume above slab
    z_min, z_max = 0.0, 3.0e-3
    Nz = 40
    Zg = np.linspace(z_min, z_max, Nz)

    focus = (0.5 * (Nx - 1) * dx, 0.5 * (Ny - 1) * dy)
    amp, phi = deep_bowl_pattern(Nx, Ny, dx, dy, lambda_, focus)

    print("Computing 3D field volume (this may take a bit)...")
    I_vol = field_3d_volume(Xg, Yg, Zg, x_emit, y_emit, amp, phi, lambda_)

    # Simple power estimate
    avg_amp2 = np.mean(amp**2)
    est_power = avg_amp2 * 0.06 * len(amp)
    print(f"Estimated slab power: {est_power:.2f} W")

    obj, mask = build_voxel_object(I_vol, Xg, Yg, Zg, frac=0.35)

    print("\n=== Holagraph v1.9 3D Voxel Solid Report ===")
    if obj is None:
        print("No 3D object detected at chosen threshold.")
    else:
        cx, cy, cz = obj["centroid"]
        print(f"Centroid: ({cx:.4f} m, {cy:.4f} m, {cz:.6f} m)")
        print(f"X range:  {obj['x_range'][0]:.4f} m to {obj['x_range'][1]:.4f} m")
        print(f"Y range:  {obj['y_range'][0]:.4f} m to {obj['y_range'][1]:.4f} m")
        print(f"Z range:  {obj['z_range'][0]:.6f} m to {obj['z_range'][1]:.6f} m")
        print(f"Voxel count: {obj['voxel_count']}")
    print("============================================")

    # STL export
    write_voxel_stl(Xg, Yg, Zg, mask, filename="deep_bowl_v1_9_voxel_solid.stl")
    print("STL export complete: deep_bowl_v1_9_voxel_solid.stl")

    # Quick diagnostics: central z-slice + 3D voxel scatter
    mid = Nz // 2
    I_mid = I_vol[mid]

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(
        I_mid,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax1, label="Intensity")
    ax1.set_title("v1.9 Central z-Slice Intensity")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_subplot(2, 2, 2)
    mask_mid = mask[mid]
    ax2.imshow(
        mask_mid,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='Greens',
        aspect='equal'
    )
    ax2.set_title("High-Intensity Mask (Central Slice)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    ax3 = fig.add_subplot(2, 2, 3)
    # projection of voxels
    iz, iy, ix = np.where(mask)
    ax3.hist2d(
        Xg[0, ix], Yg[iy, 0],
        bins=40,
        range=[[Xg.min(), Xg.max()], [Yg.min(), Yg.max()]],
        cmap='Blues'
    )
    ax3.set_title("Voxel Projection (x, y)")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    xs = Xg[0, ix]
    ys = Yg[iy, 0]
    zs = Zg[iz]
    sc = ax4.scatter(xs, ys, zs, s=4, c=zs, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, ax=ax4, label="z (m)")
    ax4.set_title("3D High-Intensity Voxels (Object Shape)")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_zlabel("z (m)")

    plt.tight_layout()
    plt.savefig("holagraph_v1_9_voxel_solid.png")
    plt.show()

    print("Holagraph v1.9 run complete. Output saved to holagraph_v1_9_voxel_solid.png")


if __name__ == "__main__":
    run_v1_9()
