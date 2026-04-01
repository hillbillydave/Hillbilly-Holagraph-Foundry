#!/usr/bin/env python3
"""
holagraph_v1_3_full.py

Holagraph v1.3 – full 3D slab with object identification:

- 2D emitter grid (Nx, Ny)
- Multiple patterns:
    * single_spot
    * two_spot
    * line_pattern
    * moving_spot (time-dependent)
- Time-dependent pattern schedule
- 3D bead simulation:
    * x, y forces from ∂I/∂x, ∂I/∂y
    * z restoring force + diffusion
- Stable, clamped dynamics
- Power accounting:
    * per-emitter relative power ~ amplitude^2
    * total power vs time
- Object identification:
    * threshold final intensity
    * label connected high-intensity regions
    * report centroids and extents
    * visualize object as 3D surface + mask overlay

NO hardware, NO serial, NO camera.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import label, center_of_mass


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
# PATTERNS (AMPLITUDE + PHASE)
# ============================================================

def single_spot_pattern(Nx, Ny, dx, dy, lambda_, spot):
    k = 2 * np.pi / lambda_
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    x0, y0 = spot
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    phi = -k * r
    amp = np.ones_like(phi)

    return amp.ravel(), phi.ravel()


def two_spot_pattern(Nx, Ny, dx, dy, lambda_, spot1, spot2):
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


def line_pattern(Nx, Ny, dx, dy, lambda_, x_line):
    k = 2 * np.pi / lambda_
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    r = np.abs(X - x_line)
    phi = -k * r
    amp = np.ones_like(phi)

    return amp.ravel(), phi.ravel()


def moving_spot_pattern(Nx, Ny, dx, dy, lambda_, t, T_total):
    xs = np.arange(Nx) * dx
    ys = np.arange(Ny) * dy

    fx = 2 + (Nx - 4) * (t / T_total)
    fy = 2 + (Ny - 4) * (t / T_total)

    x0 = fx * dx
    y0 = fy * dy

    return single_spot_pattern(Nx, Ny, dx, dy, lambda_, (x0, y0))


# ============================================================
# FIELD MODEL (2D PLANE AT z0)
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
# 3D BEAD SIMULATION WITH TIME-DEPENDENT FIELD + POWER
# ============================================================

def simulate_beads_3d_time(
    x0, y0, z0_beads,
    x_emit, y_emit,
    x_grid, y_grid,
    lambda_, z_plane,
    dt, steps,
    alpha_xy, alpha_z,
    D_xy, D_z,
    z_min, z_max,
    pattern_schedule,
    power_scale_watts=0.06
):
    x = x0.copy()
    y = y0.copy()
    z = z0_beads.copy()

    x_min, x_max = x_grid.min(), x_grid.max()
    y_min, y_max = y_grid.min(), y_grid.max()
    z_mid = 0.5 * (z_min + z_max)

    power_time = []
    I_final = None
    amp_final = None
    phi_final = None

    for step in range(steps):
        t = step * dt

        amp, phi = pattern_schedule(t)

        avg_amp2 = np.mean(amp**2)
        total_power = avg_amp2 * power_scale_watts * len(amp)
        power_time.append((t, total_power))

        E = field_2d_plane(x_grid, y_grid, x_emit, y_emit, amp, phi, lambda_, z_plane)
        I = intensity(E)

        dI_dx, dI_dy = np.gradient(I, x_grid[0, :], y_grid[:, 0])
        max_g = np.max(np.sqrt(dI_dx**2 + dI_dy**2)) + 1e-12
        dI_dx /= max_g
        dI_dy /= max_g

        dI_dx_beads = bilinear_interp(x, y, x_grid, y_grid, dI_dx)
        dI_dy_beads = bilinear_interp(x, y, x_grid, y_grid, dI_dy)

        Fx = 0.5 * alpha_xy * dI_dx_beads
        Fy = 0.5 * alpha_xy * dI_dy_beads
        Fz = -alpha_z * (z - z_mid)

        x += Fx * dt + np.sqrt(2 * D_xy * dt) * np.random.randn(*x.shape)
        y += Fy * dt + np.sqrt(2 * D_xy * dt) * np.random.randn(*y.shape)
        z += Fz * dt + np.sqrt(2 * D_z * dt) * np.random.randn(*z.shape)

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)
        z = np.clip(z, z_min, z_max)

        I_final = I
        amp_final = amp
        phi_final = phi

    power_time = np.array(power_time)

    return x, y, z, I_final, power_time, amp_final, phi_final


# ============================================================
# PATTERN SCHEDULE (v1.3)
# ============================================================

def make_pattern_schedule(Nx, Ny, dx, dy, lambda_, T_total):
    spot_single = (4 * dx, 4 * dy)
    spot1 = (4 * dx, 4 * dy)
    spot2 = (12 * dx, 12 * dy)
    x_line = 8 * dx

    def schedule(t):
        tau = t / T_total
        if tau < 0.25:
            amp, phi = single_spot_pattern(Nx, Ny, dx, dy, lambda_, spot_single)
        elif tau < 0.5:
            amp, phi = two_spot_pattern(Nx, Ny, dx, dy, lambda_, spot1, spot2)
        elif tau < 0.75:
            amp, phi = line_pattern(Nx, Ny, dx, dy, lambda_, x_line)
        else:
            amp, phi = moving_spot_pattern(Nx, Ny, dx, dy, lambda_, t, T_total)
        return amp, phi

    return schedule


# ============================================================
# OBJECT IDENTIFICATION FROM INTENSITY
# ============================================================

def identify_objects_from_intensity(I, Xg, Yg, threshold_fraction=0.6):
    I_max = I.max()
    thresh = threshold_fraction * I_max
    mask = I >= thresh

    labeled, n_labels = label(mask)
    objects = []

    for label_id in range(1, n_labels + 1):
        region_mask = (labeled == label_id)
        if not np.any(region_mask):
            continue

        cy, cx = center_of_mass(region_mask)
        cy = float(cy)
        cx = float(cx)

        x_centroid = np.interp(cx, np.arange(Xg.shape[1]), Xg[0, :])
        y_centroid = np.interp(cy, np.arange(Yg.shape[0]), Yg[:, 0])

        ys, xs = np.where(region_mask)
        x_min = Xg[0, xs.min()]
        x_max = Xg[0, xs.max()]
        y_min = Yg[ys.min(), 0]
        y_max = Yg[ys.max(), 0]

        objects.append({
            "label": label_id,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max),
            "area_pixels": int(region_mask.sum())
        })

    return objects, mask


# ============================================================
# MAIN v1.3 RUN
# ============================================================

def run_v1_3():
    Nx, Ny = 16, 16
    dx, dy = 0.01, 0.01
    lambda_ = 850e-9
    z_plane = chamber_height()

    x_emit, y_emit = emitter_grid(Nx, Ny, dx, dy)

    x_grid_1d = np.linspace(0, (Nx - 1) * dx, 200)
    y_grid_1d = np.linspace(0, (Ny - 1) * dy, 200)
    Xg, Yg = np.meshgrid(x_grid_1d, y_grid_1d, indexing='xy')

    n_beads = 1500
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
    T_total = dt * steps

    pattern_schedule = make_pattern_schedule(Nx, Ny, dx, dy, lambda_, T_total)

    xf, yf, zf, I_final, power_time, amp_final, phi_final = simulate_beads_3d_time(
        x0, y0, z0_beads,
        x_emit, y_emit,
        Xg, Yg,
        lambda_, z_plane,
        dt, steps,
        alpha_xy, alpha_z,
        D_xy, D_z,
        z_min, z_max,
        pattern_schedule,
        power_scale_watts=0.06
    )

    objects, mask = identify_objects_from_intensity(I_final, Xg, Yg, threshold_fraction=0.6)

    print("\n=== Holagraph v1.3 Object Report ===")
    if not objects:
        print("No high-intensity objects detected at the chosen threshold.")
    else:
        for obj in objects:
            print(f"Object {obj['label']}:")
            print(f"  Centroid: ({obj['x_centroid']:.4f} m, {obj['y_centroid']:.4f} m)")
            print(f"  X range:  {obj['x_range'][0]:.4f} m to {obj['x_range'][1]:.4f} m")
            print(f"  Y range:  {obj['y_range'][0]:.4f} m to {obj['y_range'][1]:.4f} m")
            print(f"  Area (pixels): {obj['area_pixels']}")
    print("====================================\n")

    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(
        I_final,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax1, label="Intensity")
    ax1.set_title("Holagraph v1.3 Intensity Map (Final Pattern)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax1_contour = plt.subplot(2, 3, 2)
    ax1_contour.imshow(
        I_final,
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    ax1_contour.contour(
        Xg, Yg, mask,
        levels=[0.5],
        colors='cyan',
        linewidths=1.5
    )
    ax1_contour.set_title("High-Intensity Object Mask Overlay")
    ax1_contour.set_xlabel("x (m)")
    ax1_contour.set_ylabel("y (m)")

    ax2 = plt.subplot(2, 3, 3)
    h = ax2.hist2d(
        xf, yf,
        bins=40,
        range=[[Xg.min(), Xg.max()], [Yg.min(), Yg.max()]],
        cmap='Blues'
    )
    ax2.set_title("Bead Distribution Projection (x, y)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    plt.colorbar(h[3], ax=ax2, label="Bead Count")

    ax3 = plt.subplot(2, 3, 4, projection='3d')
    sc = ax3.scatter(xf, yf, zf, s=4, c=zf, cmap='viridis', alpha=0.7)
    ax3.set_title("3D Bead Cloud After Simulation")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_xlim(Xg.min(), Xg.max())
    ax3.set_ylim(Yg.min(), Yg.max())
    ax3.set_zlim(z_min, z_max)
    plt.colorbar(sc, ax=ax3, label="z (m)")

    ax4 = plt.subplot(2, 3, 5)
    t_arr = power_time[:, 0]
    P_arr = power_time[:, 1]
    ax4.plot(t_arr, P_arr, color='orange')
    ax4.set_title("Estimated Total Emitter Power vs Time")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("Power (W)")
    ax4.grid(True)

    ax5 = plt.subplot(2, 3, 6, projection='3d')
    ax5.plot_surface(
        Xg, Yg, I_final,
        cmap='magma',
        linewidth=0,
        antialiased=False,
        alpha=0.9
    )
    ax5.set_title("3D Intensity Surface (Object Shape)")
    ax5.set_xlabel("x (m)")
    ax5.set_ylabel("y (m)")
    ax5.set_zlabel("Intensity (arb.)")

    plt.tight_layout()
    plt.savefig("holagraph_v1_3_full.png")
    plt.show()

    print("Holagraph v1.3 run complete. Output saved to holagraph_v1_3_full.png")
    print(f"Average power over run: {P_arr.mean():.2f} W, peak: {P_arr.max():.2f} W")


if __name__ == "__main__":
    run_v1_3()
