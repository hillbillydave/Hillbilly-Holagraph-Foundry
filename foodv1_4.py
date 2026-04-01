#!/usr/bin/env python3
"""
holagraph_v1_4_gpu.py

Holagraph v1.4 – full 3D voxel slab (ultra resolution, option C):

- 2D emitter grid (Nx, Ny)
- Full 3D intensity volume I(x, y, z)
- GPU-accelerated with PyTorch + DirectML (AMD-friendly)
- 3D bead relaxation in that volume
- Object extraction from 3D intensity:
    * threshold
    * connected region labeling (simple)
    * centroid + extents
- Power estimate from emitter amplitudes
- Plots:
    * central z-slice intensity
    * bead projection
    * 3D intensity isosurface proxy (via high-intensity mask)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# DEVICE SELECTION (DirectML if available, else CPU)
# ============================================================

def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


# ============================================================
# GEOMETRY
# ============================================================

def emitter_grid(Nx, Ny, dx, dy):
    xs = torch.arange(Nx, dtype=torch.float32) * dx
    ys = torch.arange(Ny, dtype=torch.float32) * dy
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    return X.reshape(-1), Y.reshape(-1)


def chamber_height():
    return 1.5e-3  # 1.5 mm


# ============================================================
# PATTERN (SIMPLE TWO-SPOT FOR 3D OBJECT)
# ============================================================

def two_spot_pattern(Nx, Ny, dx, dy, lambda_, spot1, spot2, device):
    k = 2 * np.pi / lambda_
    xs = torch.arange(Nx, dtype=torch.float32, device=device) * dx
    ys = torch.arange(Ny, dtype=torch.float32, device=device) * dy
    X, Y = torch.meshgrid(xs, ys, indexing='xy')

    x1, y1 = spot1
    x2, y2 = spot2

    r1 = torch.sqrt((X - x1)**2 + (Y - y1)**2)
    r2 = torch.sqrt((X - x2)**2 + (Y - y2)**2)

    phi1 = -k * r1
    phi2 = -k * r2

    phi = 0.5 * (phi1 + phi2)
    amp = torch.ones_like(phi, device=device)

    return amp.reshape(-1), phi.reshape(-1)


# ============================================================
# 3D FIELD MODEL
# ============================================================

def field_3d_volume(x_emit, y_emit, amp, phi, lambda_,
                    x_grid, y_grid, z_grid, device):
    """
    Compute E(x,y,z) on a 3D grid using GPU.

    x_grid, y_grid, z_grid: 1D tensors
    """
    k = 2 * np.pi / lambda_

    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing='xy')
    X = X.to(device)
    Y = Y.to(device)
    Z = Z.to(device)

    E_real = torch.zeros_like(X, dtype=torch.float32, device=device)
    E_imag = torch.zeros_like(X, dtype=torch.float32, device=device)

    x_emit = x_emit.to(device)
    y_emit = y_emit.to(device)
    amp = amp.to(device)
    phi = phi.to(device)

    n_emit = x_emit.shape[0]

    for i in range(n_emit):
        dx = X - x_emit[i]
        dy = Y - y_emit[i]
        dz = Z
        r = torch.sqrt(dx**2 + dy**2 + dz**2) + 1e-9

        phase = k * r + phi[i]
        a = amp[i] / r

        E_real += a * torch.cos(phase)
        E_imag += a * torch.sin(phase)

    E = torch.complex(E_real, E_imag)
    return E


def intensity_from_field(E):
    return (E.real**2 + E.imag**2)


# ============================================================
# SIMPLE 3D OBJECT IDENTIFICATION
# ============================================================

def identify_3d_object(I, x_grid, y_grid, z_grid, threshold_fraction=0.6):
    I_max = I.max().item()
    thresh = threshold_fraction * I_max
    mask = I >= thresh

    if not mask.any():
        return None, mask

    idx = mask.nonzero(as_tuple=False)
    xs = x_grid[idx[:, 0]]
    ys = y_grid[idx[:, 1]]
    zs = z_grid[idx[:, 2]]

    x_centroid = xs.mean().item()
    y_centroid = ys.mean().item()
    z_centroid = zs.mean().item()

    x_min = xs.min().item()
    x_max = xs.max().item()
    y_min = ys.min().item()
    y_max = ys.max().item()
    z_min = zs.min().item()
    z_max = zs.max().item()

    obj = {
        "x_centroid": x_centroid,
        "y_centroid": y_centroid,
        "z_centroid": z_centroid,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "z_range": (z_min, z_max),
        "voxel_count": int(mask.sum().item())
    }

    return obj, mask


# ============================================================
# BEAD RELAXATION IN 3D INTENSITY
# ============================================================

def trilinear_sample(I, x_grid, y_grid, z_grid, xb, yb, zb, device):
    """
    Trilinear interpolation of I(x,y,z) at bead positions.
    I is [Nx, Ny, Nz], grids are 1D.
    xb, yb, zb are 1D bead positions.
    """
    Nx = x_grid.shape[0]
    Ny = y_grid.shape[0]
    Nz = z_grid.shape[0]

    def find_indices(grid, p):
        idx = torch.searchsorted(grid, p) - 1
        idx = torch.clamp(idx, 0, grid.shape[0] - 2)
        return idx

    ix = find_indices(x_grid, xb)
    iy = find_indices(y_grid, yb)
    iz = find_indices(z_grid, zb)

    x1 = x_grid[ix]
    x2 = x_grid[ix + 1]
    y1 = y_grid[iy]
    y2 = y_grid[iy + 1]
    z1 = z_grid[iz]
    z2 = z_grid[iz + 1]

    tx = (xb - x1) / (x2 - x1 + 1e-12)
    ty = (yb - y1) / (y2 - y1 + 1e-12)
    tz = (zb - z1) / (z2 - z1 + 1e-12)

    I000 = I[ix,     iy,     iz    ]
    I100 = I[ix + 1, iy,     iz    ]
    I010 = I[ix,     iy + 1, iz    ]
    I110 = I[ix + 1, iy + 1, iz    ]
    I001 = I[ix,     iy,     iz + 1]
    I101 = I[ix + 1, iy,     iz + 1]
    I011 = I[ix,     iy + 1, iz + 1]
    I111 = I[ix + 1, iy + 1, iz + 1]

    c00 = I000 * (1 - tx) + I100 * tx
    c01 = I001 * (1 - tx) + I101 * tx
    c10 = I010 * (1 - tx) + I110 * tx
    c11 = I011 * (1 - tx) + I111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    c = c0 * (1 - tz) + c1 * tz
    return c


def relax_beads_in_3d(I, x_grid, y_grid, z_grid,
                       n_beads=3000, steps=200,
                       alpha=0.2, D=1e-7, device=device):
    """
    Simple gradient-descent-like relaxation of beads in 3D intensity.
    """
    x_min, x_max = x_grid[0].item(), x_grid[-1].item()
    y_min, y_max = y_grid[0].item(), y_grid[-1].item()
    z_min, z_max = z_grid[0].item(), z_grid[-1].item()

    xb = torch.empty(n_beads, device=device).uniform_(x_min, x_max)
    yb = torch.empty(n_beads, device=device).uniform_(y_min, y_max)
    zb = torch.empty(n_beads, device=device).uniform_(z_min, z_max)

    dIx, dIy, dIz = torch.gradient(I, spacing=(x_grid, y_grid, z_grid))

    for _ in range(steps):
        gx = trilinear_sample(dIx, x_grid, y_grid, z_grid, xb, yb, zb, device)
        gy = trilinear_sample(dIy, x_grid, y_grid, z_grid, xb, yb, zb, device)
        gz = trilinear_sample(dIz, x_grid, y_grid, z_grid, xb, yb, zb, device)

        xb += alpha * gx + torch.sqrt(torch.tensor(D, device=device)) * torch.randn_like(xb)
        yb += alpha * gy + torch.sqrt(torch.tensor(D, device=device)) * torch.randn_like(yb)
        zb += alpha * gz + torch.sqrt(torch.tensor(D, device=device)) * torch.randn_like(zb)

        xb = torch.clamp(xb, x_min, x_max)
        yb = torch.clamp(yb, y_min, y_max)
        zb = torch.clamp(zb, z_min, z_max)

    return xb.cpu().numpy(), yb.cpu().numpy(), zb.cpu().numpy()


# ============================================================
# MAIN v1.4 RUN (ULTRA RESOLUTION: OPTION C)
# ============================================================

def run_v1_4():
    # Emitter grid
    Nx_emit, Ny_emit = 16, 16
    dx, dy = 0.01, 0.01
    lambda_ = 850e-9

    x_emit_cpu, y_emit_cpu = emitter_grid(Nx_emit, Ny_emit, dx, dy)
    x_emit = x_emit_cpu.to(device)
    y_emit = y_emit_cpu.to(device)

    # Pattern: two-spot
    spot1 = (4 * dx, 4 * dy)
    spot2 = (12 * dx, 12 * dy)
    amp, phi = two_spot_pattern(Nx_emit, Ny_emit, dx, dy, lambda_, spot1, spot2, device)

    # 3D voxel grid (option C: 256 x 256 x 128)
    Nx, Ny, Nz = 256, 256, 128
    x_grid = torch.linspace(0.0, (Nx_emit - 1) * dx, Nx, device=device)
    y_grid = torch.linspace(0.0, (Ny_emit - 1) * dy, Ny, device=device)
    z_grid = torch.linspace(0.0, 3.0e-3, Nz, device=device)

    print("Computing 3D field volume (this may take a bit)...")
    E = field_3d_volume(x_emit, y_emit, amp, phi, lambda_, x_grid, y_grid, z_grid, device)
    I = intensity_from_field(E)

    avg_amp2 = (amp**2).mean().item()
    power_scale_watts = 0.06
    total_power = avg_amp2 * power_scale_watts * amp.shape[0]
    print(f"Estimated slab power: {total_power:.2f} W")

    print("Identifying 3D object...")
    obj, mask = identify_3d_object(I, x_grid, y_grid, z_grid, threshold_fraction=0.6)

    print("\n=== Holagraph v1.4 3D Object Report (Ultra Resolution) ===")
    if obj is None:
        print("No high-intensity object detected at chosen threshold.")
    else:
        print(f"Centroid: ({obj['x_centroid']:.4f} m, "
              f"{obj['y_centroid']:.4f} m, {obj['z_centroid']:.6f} m)")
        print(f"X range:  {obj['x_range'][0]:.4f} m to {obj['x_range'][1]:.4f} m")
        print(f"Y range:  {obj['y_range'][0]:.4f} m to {obj['y_range'][1]:.4f} m")
        print(f"Z range:  {obj['z_range'][0]:.6f} m to {obj['z_range'][1]:.6f} m")
        print(f"Voxel count: {obj['voxel_count']}")
    print("==========================================================\n")

    print("Relaxing beads in 3D intensity...")
    xb, yb, zb = relax_beads_in_3d(I, x_grid, y_grid, z_grid,
                                   n_beads=3000, steps=200,
                                   alpha=0.2, D=1e-7, device=device)

    I_cpu = I.detach().cpu().numpy()
    xg_cpu = x_grid.detach().cpu().numpy()
    yg_cpu = y_grid.detach().cpu().numpy()
    zg_cpu = z_grid.detach().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy()

    mid_z_idx = Nz // 2
    I_slice = I_cpu[:, :, mid_z_idx]
    mask_slice = mask_cpu[:, :, mid_z_idx]

    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(
        I_slice.T,
        extent=[xg_cpu.min(), xg_cpu.max(), yg_cpu.min(), yg_cpu.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax1, label="Intensity")
    ax1.set_title("v1.4 Central z-Slice Intensity")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(
        mask_slice.T,
        extent=[xg_cpu.min(), xg_cpu.max(), yg_cpu.min(), yg_cpu.max()],
        origin='lower',
        cmap='Greens',
        aspect='equal'
    )
    ax2.set_title("High-Intensity Mask (Central Slice)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    ax3 = plt.subplot(2, 2, 3)
    h = ax3.hist2d(
        xb, yb,
        bins=60,
        range=[[xg_cpu.min(), xg_cpu.max()], [yg_cpu.min(), yg_cpu.max()]],
        cmap='Blues'
    )
    plt.colorbar(h[3], ax=ax3, label="Bead Count")
    ax3.set_title("Bead Projection (x, y)")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = plt.subplot(2, 2, 4, projection='3d')
    idx_mask = np.where(mask_cpu)
    xs = xg_cpu[idx_mask[0]]
    ys = yg_cpu[idx_mask[1]]
    zs = zg_cpu[idx_mask[2]]
    ax4.scatter(xs, ys, zs, s=1, c=zs, cmap='plasma', alpha=0.5)
    ax4.set_title("3D High-Intensity Voxels (Object Shape)")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_zlabel("z (m)")

    plt.tight_layout()
    plt.savefig("holagraph_v1_4_gpu_ultra.png")
    plt.show()

    print("Holagraph v1.4 run complete. Output saved to holagraph_v1_4_gpu_ultra.png")


if __name__ == "__main__":
    run_v1_4()

