"""
Holagraph Media Framework — Ethical License and Use Conditions
Author: David E. Blackwell (2026)

This software and its underlying physical framework are released freely for:
    - scientific research
    - artistic and media development
    - exploratory spacecraft interfaces
    - educational use
    - peaceful technological advancement

The author places the following conditions on its use:

1. PEACEFUL USE ONLY
   This framework may NOT be used for:
       - weapons
       - military targeting systems
       - harmful surveillance
       - any application intended to injure, coerce, or intimidate

2. FREE FRAMEWORK, BUT PROTOTYPE REQUIRED
   If any commercial holagraph media system is built from this work,
   the author requests ONE WORKING PROTOTYPE UNIT at no cost.
   This is not a financial claim; it is recognition of authorship.

3. PHYSICS GROUNDING REQUIREMENT
   Any implementation must remain grounded in real physics.
   Assumptions, approximations, and scaling factors must be documented.
   Speculative features must not be misrepresented as proven capabilities.

4. RADIATION SHIELDING DISCLAIMER
   Reinforced optical slabs are NOT radiation shields for:
       - solar particle events
       - galactic cosmic rays
       - ionizing radiation
   These require mass shielding or magnetic/plasma systems.

5. ETHICAL INTENT
   This framework is released to improve the world, support peaceful exploration,
   and expand media and scientific expression. Misuse violates the intent of the work.

By using this software, you agree to these conditions.
"""

import numpy as np

# ============================================================
# 0. MASTER SETTINGS
# ============================================================

TILE_SIZE = 0.1
TILE_AREA = TILE_SIZE**2
SLAB_Z = 0.05

NX = NY = 16
WAVELENGTH = 1e-6

# Optical target intensity
I_TARGET_MEAN = 10.0

# Phase learning
PHASE_LR = 1e-3
PHASE_LR_MIN = 1e-6
PHASE_LR_MAX = 5e-4

# Time stepping
STEPS = 400
DT = 1.0  # arbitrary time unit per step

# Multi-layer slab: soft + rigid
S_SOFT_MIN = 0.5
S_SOFT_MAX = 5.0
S_RIGID_MIN = 1.5
S_RIGID_MAX = 20.0

# Reinforcement gains (soft vs rigid)
K_S_SOFT = 0.05
K_P_SOFT = 0.001
K_S_RIGID = 0.1
K_P_RIGID = 0.0005

# Time-dependent reinforcement (shock absorption)
ALPHA_SOFT = 0.2   # response speed of soft layer
ALPHA_RIGID = 0.1  # response speed of rigid layer

# Energy budgeting
P_MAX = 800.0   # max allowed "arb" power before throttling

# Thermal model
T_AMB = 300.0   # ambient temperature (K)
T = T_AMB
C_TH = 1.0      # heat capacity (arb)
R_COOL = 0.01   # cooling rate
T_HOT = 340.0   # threshold where performance starts to drop
THERMAL_SOFTEN = 0.5  # factor reducing rigid reinforcement when hot

# Amplitude limits
A_MAX = 20.0
GRAD_CLIP = 500

# ============================================================
# 1. EMITTER GEOMETRY
# ============================================================

N = NX * NY
d = TILE_SIZE / NX

xs = np.linspace(d/2, TILE_SIZE - d/2, NX)
ys = np.linspace(d/2, TILE_SIZE - d/2, NY)
emitters = np.array([(x, y, 0.0) for x in xs for y in ys])

# ============================================================
# 2. CONTROL POINTS
# ============================================================

Mx = My = 16
xs_c = np.linspace(d/2, TILE_SIZE - d/2, Mx)
ys_c = np.linspace(d/2, TILE_SIZE - d/2, My)
controls = np.array([(x, y, SLAB_Z) for x in xs_c for y in ys_c])
M = len(controls)

# ============================================================
# 3. GREEN'S MATRIX
# ============================================================

k = 2*np.pi / WAVELENGTH
G = np.zeros((M, N), dtype=complex)

for m, (xm, ym, zm) in enumerate(controls):
    for n, (xn, yn, zn) in enumerate(emitters):
        r = np.sqrt((xm-xn)**2 + (ym-yn)**2 + (zm-zn)**2)
        G[m, n] = np.exp(1j*k*r) / r

# ============================================================
# 4. INITIAL EMITTER STATE (SEED FOCUS)
# ============================================================

cx, cy = TILE_SIZE/2, TILE_SIZE/2
phi = np.zeros(N)

for i, (x, y, z) in enumerate(emitters):
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx*dx + dy*dy + SLAB_Z*SLAB_Z)
    phi[i] = k * r

phi += np.random.uniform(-0.1, 0.1, N)

# Spatially varying base shape (slightly softer at edges)
A_shape = np.ones(N)
for i, (x, y, z) in enumerate(emitters):
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    A_shape[i] = 1.0 - 0.3 * (r / (TILE_SIZE/2))  # softer toward edges
A_shape = np.clip(A_shape, 0.5, 1.0)

# Multi-layer S fields (spatially varying)
S_soft = np.full(N, S_SOFT_MIN)
S_rigid = np.full(N, S_RIGID_MIN)

# ============================================================
# 5. FORCE MODEL
# ============================================================

def measured_force(I_mean, delta):
    # Effective mechanical stiffness (soft + rigid)
    k_eff_soft = 2e5
    k_eff_rigid = 5e5
    F_mech = (k_eff_soft + k_eff_rigid) * delta
    F_em = (TILE_AREA / (2 * 3e8)) * I_mean
    return F_mech + F_em

def desired_force(delta):
    # Strong reinforced target force
    return 20000 * delta

# Dynamic DELTA(t): impact + oscillation
def delta_t(t):
    delta = 0.0
    if 100 <= t < 130:
        delta += 0.02  # impact pulse
    delta += 0.005 * np.sin(2 * np.pi * t / 80.0)  # background oscillation
    return delta

# ============================================================
# 6. SIMULATION LOOP
# ============================================================

power_log = []
I_mean_log = []
T_log = []

for t in range(STEPS):

    # Effective S field and amplitude
    S_eff = S_soft + S_rigid
    A = np.clip(S_eff, 0.0, A_MAX) * A_shape

    # --- Field ---
    a = A * np.exp(1j * phi)
    E = G @ a
    I = np.abs(E)**2
    I_mean = np.mean(I)

    if t % 50 == 0:
        print(f"step {t}: I_mean={I_mean}, I_min={I.min()}, I_max={I.max()}")

    # --- Phase gradient ---
    grad_phi = np.zeros(N)
    for n in range(N):
        term = 0.0
        for m in range(M):
            dI_dphi = 2*np.imag(G[m,n] * A[n] * np.exp(1j*phi[n]) * np.conj(E[m]))
            term += 4*(I[m] - I_TARGET_MEAN) * dI_dphi
        grad_phi[n] = term

    grad_phi = np.clip(grad_phi, -GRAD_CLIP, GRAD_CLIP)

    # Adaptive phase learning
    if np.max(np.abs(grad_phi)) > 0.5 * GRAD_CLIP:
        PHASE_LR = max(PHASE_LR * 0.5, PHASE_LR_MIN)
    else:
        PHASE_LR = min(PHASE_LR * 1.01, PHASE_LR_MAX)

    phi -= PHASE_LR * grad_phi

    # --- Dynamic displacement ---
    DELTA = delta_t(t)

    # --- Force and reinforcement ---
    Fm = measured_force(I_mean, DELTA)
    Fd = desired_force(DELTA)
    eF = Fd - Fm

    # Target updates for soft and rigid layers
    S_soft_target = S_soft + K_S_SOFT * eF - K_P_SOFT * S_soft
    S_rigid_target = S_rigid + K_S_RIGID * eF - K_P_RIGID * S_rigid

    # Thermal softening: when hot, rigid layer is less aggressive
    if T > T_HOT:
        S_rigid_target = S_rigid + THERMAL_SOFTEN * (S_rigid_target - S_rigid)

    # Energy budgeting: throttle reinforcement if over power budget
    P_inst = np.sum(A**2)
    if P_inst > P_MAX:
        scale = P_MAX / (P_inst + 1e-9)
        S_soft_target = S_soft + scale * (S_soft_target - S_soft)
        S_rigid_target = S_rigid + scale * (S_rigid_target - S_rigid)

    # Time-dependent reinforcement (shock absorption)
    S_soft = S_soft + ALPHA_SOFT * (S_soft_target - S_soft)
    S_rigid = S_rigid + ALPHA_RIGID * (S_rigid_target - S_rigid)

    # Clamp layers
    S_soft = np.clip(S_soft, S_SOFT_MIN, S_SOFT_MAX)
    S_rigid = np.clip(S_rigid, S_RIGID_MIN, S_RIGID_MAX)

    # --- Thermal update ---
    T += DT * (P_inst - R_COOL * (T - T_AMB)) / C_TH

    power_log.append(P_inst)
    I_mean_log.append(I_mean)
    T_log.append(T)

print("Simulation complete.")
print("Peak power (arb units):", max(power_log))
print("Final mean intensity (sim units):", I_mean_log[-1])
print("Final temperature (arb K):", T)
print("Final phase LR:", PHASE_LR)
print("Soft layer S range:", float(S_soft.min()), "to", float(S_soft.max()))
print("Rigid layer S range:", float(S_rigid.min()), "to", float(S_rigid.max()))

# ============================================================
# 7. PHYSICAL POWER ESTIMATE (MEDIUM-POWER PROTOTYPE)
# ============================================================

# Calibration: when I_mean_sim = 1e5, we want I_phys = 1e4 W/m^2
# => tile power ~ 100 W at that operating point (medium-power regime)
I_sim_ref = 1e5       # reference simulated intensity
I_phys_ref = 1e4      # W/m^2 corresponding physical intensity
k_I = I_phys_ref / I_sim_ref

I_mean_final = I_mean_log[-1]
I_phys_mean = k_I * I_mean_final
P_watts = I_phys_mean * TILE_AREA

print("Calibrated physical mean intensity (W/m^2):", I_phys_mean)
print("Estimated tile power (watts):", P_watts)
