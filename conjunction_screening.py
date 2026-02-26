"""
conjunction_screening.py
========================
Demonstrate real-time LEO conjunction screening using the trained Fourier-PINN
as a fast surrogate propagator.

For circular coplanar orbits at the same altitude, satellites maintain their
relative phase -- so we use a MIXED-altitude scenario:
  - Catalog A (N_A satellites): 400 km circular orbit (our trained PINN)
  - Catalog B (N_B satellites): 405 km circular orbit (slightly slower)

Because the orbits have different periods, satellites drift relative to each
other and undergo periodic close approaches.

Key demonstration:
  - Batch-propagate ALL satellites in a single PINN forward pass
  - KD-tree spatial indexing for O(N log N) conjunction screening
  - Compare KD-tree vs brute-force O(N²) screening with scalability benchmark
  - Compare PINN propagation timing against N individual scipy ODE integrations

Outputs
-------
    figures/conjunction_screening.png  -- 4-panel summary figure
    figures/conjunction_table.txt      -- timing and detection summary
"""

import os
import sys
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.physics import MU, R_EARTH, R_ORBIT, NormalizationParams
from src.models import FourierPINN

# ── Config ────────────────────────────────────────────────────────────────────
N_A          = 50          # satellites in catalog A (400 km)
N_B          = 50          # satellites in catalog B (405 km)
N_TIME       = 1000        # timesteps for propagation
CONJUNC_KM   = 10.0        # conjunction threshold (km)
R_B          = R_ORBIT + 5.0   # 405 km altitude

FIGURES_DIR  = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ── Load PINN ─────────────────────────────────────────────────────────────────
# Uses FourierPINN from src/models.py (N_FREQ=8, secular drift head).
print("Loading PINN model...")
norm_A = NormalizationParams(r_ref=R_ORBIT)
model  = FourierPINN().double()
sd     = torch.load("models/pinn_j2.pt", map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=False)   # sec_head may be absent in older checkpoints (zero-init default)
model.eval()

# ── Orbital parameters ────────────────────────────────────────────────────────
v_A   = math.sqrt(MU / R_ORBIT)      # km/s circular velocity at 400 km
v_B   = math.sqrt(MU / R_B)          # km/s at 405 km
T_A   = 2 * math.pi * math.sqrt(R_ORBIT**3 / MU)  # period A (s)
T_B   = 2 * math.pi * math.sqrt(R_B**3     / MU)  # period B (s)

# Propagate for 3 full periods of the longer orbit
T_SIM = 3 * T_B
t_vec = np.linspace(0, T_SIM, N_TIME)   # (N_TIME,)  seconds

print(f"Orbit A: r={R_ORBIT:.1f} km, T={T_A/60:.2f} min, v={v_A:.4f} km/s")
print(f"Orbit B: r={R_B:.1f}  km, T={T_B/60:.2f} min, v={v_B:.4f} km/s")
print(f"Simulation span: {T_SIM/3600:.2f} h  ({N_TIME} timesteps)")

# Phase offsets: uniformly distributed around the orbit
phi_A = np.linspace(0, 2*math.pi, N_A, endpoint=False)   # radians
phi_B = np.linspace(0, 2*math.pi, N_B, endpoint=False)

# ── PINN batch propagation (Catalog A only -- trained on this orbit) ───────────
# For a circular orbit, satellite with phase offset φ is equivalent to the
# base satellite queried at time t + φ/n (where n is mean motion).
# We exploit this to propagate N_A satellites with a single forward pass.

print(f"\nPINN batch propagation: {N_A} satellites × {N_TIME} timesteps...")
t0_pinn = time.perf_counter()

n_A = 2 * math.pi / T_A   # mean motion [rad/s]

# For each satellite i and time step j: t_query = t_vec[j] + phi_A[i] / n_A
# Shape: (N_A, N_TIME)
t_queries_A = (t_vec[None, :] + (phi_A / n_A)[:, None])  # seconds

# Flatten, normalize, forward, un-normalize
t_flat_norm = (t_queries_A.reshape(-1, 1) / norm_A.t_ref)
t_tensor    = torch.tensor(t_flat_norm, dtype=torch.float64)

with torch.no_grad():
    pos_norm = model(t_tensor)               # (N_A*N_TIME, 3) normalized

pos_A_pinn = pos_norm.numpy() * norm_A.r_ref  # km, shape (N_A*N_TIME, 3)
pos_A_pinn = pos_A_pinn.reshape(N_A, N_TIME, 3)

t_pinn_ms = (time.perf_counter() - t0_pinn) * 1e3
print(f"  PINN inference done in {t_pinn_ms:.1f} ms")

# ── ODE integration (Catalog A, for timing comparison) ────────────────────────
def two_body_ode(t, s):
    x, y, z, vx, vy, vz = s
    r3 = (x*x + y*y + z*z) ** 1.5
    return [vx, vy, vz, -MU*x/r3, -MU*y/r3, -MU*z/r3]

print(f"\nODE integration: {N_A} satellites × {N_TIME} timesteps...")
t0_ode = time.perf_counter()
pos_A_ode = np.zeros((N_A, N_TIME, 3))
for i, phi in enumerate(phi_A):
    s0 = [R_ORBIT * math.cos(phi), R_ORBIT * math.sin(phi), 0.0,
          -v_A * math.sin(phi),    v_A * math.cos(phi),     0.0]
    sol = solve_ivp(two_body_ode, [0, T_SIM], s0,
                    t_eval=t_vec, method="DOP853",
                    rtol=1e-9, atol=1e-9)
    pos_A_ode[i] = sol.y[:3].T

t_ode_ms = (time.perf_counter() - t0_ode) * 1e3
print(f"  ODE integration done in {t_ode_ms:.0f} ms")
print(f"  Speed-up: {t_ode_ms / t_pinn_ms:.1f}x faster with PINN")

# ── Catalog B: analytical (perfect circular) for demonstration ────────────────
# For Catalog B (different altitude) we use the analytical circular solution.
# In a full system, B would use its own trained PINN.
n_B = 2 * math.pi / T_B
pos_B = np.zeros((N_B, N_TIME, 3))
for i, phi in enumerate(phi_B):
    angles = n_B * t_vec + phi
    pos_B[i, :, 0] = R_B * np.cos(angles)
    pos_B[i, :, 1] = R_B * np.sin(angles)
    pos_B[i, :, 2] = 0.0

# ── PINN accuracy vs ODE ───────────────────────────────────────────────────────
pinn_err = np.linalg.norm(pos_A_pinn - pos_A_ode, axis=2)   # (N_A, N_TIME)
mean_err = float(np.mean(pinn_err))
max_err  = float(np.max(pinn_err))
print(f"\nPINN vs ODE accuracy over {N_A} satellites:")
print(f"  Mean error: {mean_err:.2f} km")
print(f"  Max  error: {max_err:.2f} km")

# ── Conjunction screening ─────────────────────────────────────────────────────

# --- Method 1: Brute-force O(N_A × N_B × T) ---
print(f"\nBrute-force screening: {N_A * N_B:,} pairs × {N_TIME} timesteps...")
t0_brute = time.perf_counter()

diff      = pos_A_pinn[:, None, :, :] - pos_B[None, :, :, :]   # (N_A, N_B, N_TIME, 3)
distances = np.linalg.norm(diff, axis=3)                         # (N_A, N_B, N_TIME)
min_dist  = distances.min(axis=2)                                 # (N_A, N_B)

t_brute_ms = (time.perf_counter() - t0_brute) * 1e3
n_conj_brute = int((min_dist < CONJUNC_KM).sum())
overall_min  = float(min_dist.min())
print(f"  Done in {t_brute_ms:.1f} ms  |  Conjunctions: {n_conj_brute}")

# --- Method 2: KD-tree O((N_A + N_B) × log(N_A + N_B) × T) ---
print(f"KD-tree screening: {N_A + N_B} satellites × {N_TIME} timesteps...")
t0_kdtree = time.perf_counter()

conj_events = []
for t_idx in range(N_TIME):
    all_pos = np.vstack([pos_A_pinn[:, t_idx, :], pos_B[:, t_idx, :]])
    tree = cKDTree(all_pos)
    pairs = tree.query_pairs(r=CONJUNC_KM)
    for (i, j) in pairs:
        if i < N_A and j >= N_A:
            conj_events.append((i, j - N_A, t_idx))
        elif j < N_A and i >= N_A:
            conj_events.append((j, i - N_A, t_idx))

t_kdtree_ms = (time.perf_counter() - t0_kdtree) * 1e3
conj_pairs_kd = set((ev[0], ev[1]) for ev in conj_events)
n_conj_kdtree = len(conj_pairs_kd)
print(f"  Done in {t_kdtree_ms:.1f} ms  |  Conjunctions: {n_conj_kdtree}")
print(f"  Close-approach events across all timesteps: {len(conj_events)}")

# Validate both methods agree
assert n_conj_brute == n_conj_kdtree, \
    f"Mismatch: brute={n_conj_brute}, KD-tree={n_conj_kdtree}"
print(f"  Methods agree: {n_conj_brute} conjunction pairs, closest {overall_min:.2f} km")

n_conjunctions = n_conj_brute   # used by plots / summary

# ── Scalability benchmark (single-timestep, synthetic positions) ──────────────
print("\n-- Scalability: Brute-force vs KD-tree (single timestep) --")
scale_results = []
for N_test in [100, 500, 1000, 2000]:
    rng = np.random.default_rng(42)
    r_rand = R_ORBIT + rng.uniform(-10, 10, N_test)
    theta_rand = rng.uniform(0, 2 * math.pi, N_test)
    test_pos = np.column_stack([
        r_rand * np.cos(theta_rand),
        r_rand * np.sin(theta_rand),
        rng.uniform(-50, 50, N_test),
    ])
    # Brute-force
    t0 = time.perf_counter()
    diffs = test_pos[:, None, :] - test_pos[None, :, :]
    _ = int((np.linalg.norm(diffs, axis=2) < CONJUNC_KM).sum()) // 2
    t_bf = (time.perf_counter() - t0) * 1e3
    del diffs
    # KD-tree
    t0 = time.perf_counter()
    _ = len(cKDTree(test_pos).query_pairs(r=CONJUNC_KM))
    t_kd = (time.perf_counter() - t0) * 1e3
    scale_results.append((N_test, t_bf, t_kd))
    ratio = t_bf / t_kd if t_kd > 0.01 else float("inf")
    print(f"  N={N_test:>5}: Brute={t_bf:>8.1f} ms  KD-tree={t_kd:>8.1f} ms  ({ratio:.1f}x)")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"Real-Time LEO Conjunction Screening via Fourier-PINN\n"
    f"({N_A} Catalog-A + {N_B} Catalog-B satellites, 3-orbit window)",
    fontsize=12, y=1.01
)

# Panel 1: Satellite positions at t=0 (top-down)
ax = axes[0, 0]
theta = np.linspace(0, 2*math.pi, 300)
ax.plot(R_ORBIT * np.cos(theta), R_ORBIT * np.sin(theta),
        "b--", lw=0.8, alpha=0.5, label=f"Orbit A ({R_ORBIT:.0f} km)")
ax.plot(R_B * np.cos(theta), R_B * np.sin(theta),
        "g--", lw=0.8, alpha=0.5, label=f"Orbit B ({R_B:.0f} km)")
ax.scatter(pos_A_pinn[:, 0, 0], pos_A_pinn[:, 0, 1],
           c="tab:blue", s=20, zorder=3, label="Catalog A (PINN)")
ax.scatter(pos_B[:, 0, 0],    pos_B[:, 0, 1],
           c="tab:green", s=20, zorder=3, label="Catalog B")
earth = plt.Circle((0, 0), R_EARTH, color="cornflowerblue", alpha=0.4)
ax.add_patch(earth)
ax.set_aspect("equal")
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.set_title("Initial Satellite Distribution (t = 0)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 2: PINN vs ODE error over time
ax = axes[0, 1]
for i in range(min(10, N_A)):
    ax.plot(t_vec / 3600, pinn_err[i], lw=0.6, alpha=0.6)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("PINN vs ODE Error (km)")
ax.set_title(f"PINN Accuracy: {N_A} Satellites (sample of 10 shown)")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.text(0.97, 0.95, f"Mean: {mean_err:.1f} km\nMax: {max_err:.1f} km",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), fontsize=9)

# Panel 3: Minimum pairwise distance heatmap (first 20x20 pairs)
ax = axes[1, 0]
n_show = min(20, N_A)
img = ax.imshow(min_dist[:n_show, :n_show], aspect="auto", origin="lower",
                cmap="RdYlGn", vmin=0, vmax=50)
fig.colorbar(img, ax=ax, label="Min. separation (km)")
ax.set_xlabel("Catalog-B satellite index")
ax.set_ylabel("Catalog-A satellite index")
ax.set_title(f"Min. A–B Separation Heatmap ({n_show}×{n_show} shown)")
# Overlay threshold contour
cs = ax.contour(min_dist[:n_show, :n_show], levels=[CONJUNC_KM], colors="red", linewidths=1.5)
ax.clabel(cs, fmt=f"{CONJUNC_KM:.0f} km threshold")

# Panel 4: Timing comparison (propagation + screening)
ax = axes[1, 1]
methods  = ["PINN\npropag.", "KD-tree\nscreen", "Brute-force\nscreen", "ODE\npropag."]
times_ms = [t_pinn_ms, t_kdtree_ms, t_brute_ms, t_ode_ms]
cols     = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
bars     = ax.bar(methods, times_ms, color=cols, alpha=0.85, edgecolor="k", lw=0.8)
for bar, v in zip(bars, times_ms):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times_ms) * 0.02,
            f"{v:.0f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Wall-clock time (ms)")
ax.set_title(f"Timing: {N_A + N_B} Satellites × {N_TIME} Timesteps")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_png = os.path.join(FIGURES_DIR, "conjunction_screening.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Text summary ───────────────────────────────────────────────────────────────
table_path = os.path.join(FIGURES_DIR, "conjunction_table.txt")
with open(table_path, "w") as f:
    f.write("LEO Conjunction Screening -- PINN Surrogate Propagator\n")
    f.write("=" * 64 + "\n\n")
    f.write(f"Catalog A: {N_A} satellites @ {R_ORBIT:.1f} km (400 km LEO)\n")
    f.write(f"Catalog B: {N_B} satellites @ {R_B:.1f} km  (405 km LEO)\n")
    f.write(f"Time window: {T_SIM/3600:.2f} h ({N_TIME} timesteps)\n\n")
    f.write(f"{'Metric':<40}  {'Value':>18}\n")
    f.write("-" * 64 + "\n")
    f.write(f"{'PINN batch inference time':<40}  {t_pinn_ms:>15.1f} ms\n")
    f.write(f"{'ODE integration time':<40}  {t_ode_ms:>15.0f} ms\n")
    f.write(f"{'PINN propagation speed-up':<40}  {t_ode_ms/t_pinn_ms:>14.1f} x\n")
    f.write(f"{'Brute-force screening time':<40}  {t_brute_ms:>15.1f} ms\n")
    f.write(f"{'KD-tree screening time':<40}  {t_kdtree_ms:>15.1f} ms\n")
    if t_kdtree_ms > 0.01:
        f.write(f"{'KD-tree vs brute-force':<40}  {t_brute_ms/t_kdtree_ms:>14.1f} x\n")
    f.write(f"{'Pairs screened':<40}  {N_A*N_B:>18,}\n")
    f.write(f"{'Conjunctions (< {CONJUNC_KM} km)':<40}  {n_conjunctions:>18}\n")
    f.write(f"{'Close-approach events':<40}  {len(conj_events):>18,}\n")
    f.write(f"{'Closest approach':<40}  {overall_min:>15.2f} km\n")
    f.write(f"{'PINN mean error vs ODE':<40}  {mean_err:>15.2f} km\n")
    f.write(f"{'PINN max  error vs ODE':<40}  {max_err:>15.2f} km\n")
    f.write("=" * 64 + "\n\n")
    f.write("Scalability Benchmark (single timestep)\n")
    f.write("-" * 64 + "\n")
    f.write(f"{'N satellites':<15} {'Brute-force':>15} {'KD-tree':>15} {'Speed-up':>12}\n")
    f.write("-" * 64 + "\n")
    for N_s, t_bf, t_kd in scale_results:
        ratio = t_bf / t_kd if t_kd > 0.01 else float("inf")
        f.write(f"{N_s:<15} {t_bf:>13.1f} ms {t_kd:>13.1f} ms {ratio:>10.1f} x\n")
    f.write("=" * 64 + "\n")
print(f"Saved: {table_path}")
print("\nDone.")
