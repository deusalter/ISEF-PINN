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
    figures/conjunction_screening.png  -- 6-panel summary figure
    figures/conjunction_table.txt      -- timing, detection, and MC Dropout summary
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

MC_SAMPLES   = 50          # MC Dropout forward passes
DROPOUT_P    = 0.1          # dropout probability for MC mode
R_COMBINED   = 0.010        # combined hard-body radius (km) -- ~10 m

# ── Chan (2008) collision probability ─────────────────────────────────────────
def collision_probability_chan2008(pos_A, pos_B, cov_A, cov_B, r_combined):
    """Short-encounter collision probability (Chan 2008, Eq. 3).

    Parameters
    ----------
    pos_A, pos_B : (3,) arrays -- positions at TCA (km)
    cov_A, cov_B : (3,3) arrays -- 3D position covariance (km^2)
    r_combined   : float -- combined hard-body radius (km)

    Returns
    -------
    Pc : float -- collision probability
    """
    dr = pos_A - pos_B                          # relative position (3,)
    C = cov_A + cov_B                           # combined covariance (3,3)
    miss = np.linalg.norm(dr)
    if miss < 1e-12:
        return 1.0

    # Build encounter-plane basis perpendicular to miss vector
    e_miss = dr / miss
    # Find a vector not parallel to e_miss
    seed = np.array([1.0, 0.0, 0.0]) if abs(e_miss[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e_miss, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e_miss, e1)
    P = np.column_stack([e1, e2])               # (3, 2) projection matrix

    # Project to 2D encounter plane
    dr_2d = P.T @ dr                            # (2,)
    C_2d = P.T @ C @ P                          # (2, 2)

    det_C = np.linalg.det(C_2d)
    if det_C < 1e-30:
        return 0.0

    C_inv = np.linalg.inv(C_2d)
    maha = float(dr_2d @ C_inv @ dr_2d)
    Pc = (r_combined**2 / (2.0 * np.sqrt(det_C))) * np.exp(-0.5 * maha)
    return float(np.clip(Pc, 0.0, 1.0))

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

# ── MC Dropout Uncertainty Quantification ─────────────────────────────────────
# NOTE: The PINN was trained WITHOUT dropout.  Applying dropout at inference
# gives an *approximate* posterior that is illustrative, not calibrated.  A
# production system would retrain with dropout enabled.

print("\n-- MC Dropout uncertainty quantification --")
mc_model = FourierPINN(dropout_p=DROPOUT_P).double()
mc_model.load_state_dict(sd, strict=False)
mc_model.eval()          # batchnorm/etc in eval; F.dropout overrides per-pass
mc_model.enable_dropout()

# Group conjunction events by (sat_A, sat_B) and find TCA (time of closest approach)
from collections import defaultdict
pair_events = defaultdict(list)
for (iA, iB, t_idx) in conj_events:
    pair_events[(iA, iB)].append(t_idx)

conj_results = []   # will hold dicts for each pair
for (iA, iB), t_indices in sorted(pair_events.items()):
    # TCA = timestep with minimum distance for this pair
    dists_at_times = np.array([
        np.linalg.norm(pos_A_pinn[iA, ti] - pos_B[iB, ti]) for ti in t_indices
    ])
    tca_idx = t_indices[np.argmin(dists_at_times)]
    miss_km = float(np.min(dists_at_times))

    # MC forward for satellite A at TCA
    t_query_A = t_vec[tca_idx] + phi_A[iA] / n_A   # seconds
    t_norm_val = t_query_A / norm_A.t_ref
    t_mc = torch.tensor([[t_norm_val]], dtype=torch.float64)

    with torch.no_grad():
        samples = mc_model.mc_forward(t_mc, n_mc=MC_SAMPLES)  # (MC, 1, 3)
    samples_km = samples[:, 0, :].numpy() * norm_A.r_ref       # (MC, 3) km

    cov_A = np.cov(samples_km.T)                                # (3, 3)
    sigma_xyz = np.sqrt(np.diag(cov_A))                         # per-axis 1-sigma

    # Catalog B is analytical -> zero covariance
    cov_B = np.zeros((3, 3))

    Pc = collision_probability_chan2008(
        pos_A_pinn[iA, tca_idx], pos_B[iB, tca_idx],
        cov_A, cov_B, R_COMBINED
    )

    conj_results.append({
        "iA": iA, "iB": iB, "tca_idx": tca_idx,
        "miss_km": miss_km, "sigma": sigma_xyz, "Pc": Pc, "cov_A": cov_A,
    })

mc_model.disable_dropout()

print(f"  Evaluated {len(conj_results)} conjunction pairs with MC Dropout (n_mc={MC_SAMPLES})")
if conj_results:
    Pc_vals = [r["Pc"] for r in conj_results]
    print(f"  Pc range: [{min(Pc_vals):.2e}, {max(Pc_vals):.2e}]")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
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

# Panel 5: Miss distance vs position 1-sigma (MC Dropout)
ax = axes[2, 0]
if conj_results:
    miss_vals = [r["miss_km"] for r in conj_results]
    sigma_vals = [np.mean(r["sigma"]) for r in conj_results]
    ax.scatter(sigma_vals, miss_vals, c="tab:purple", s=30, alpha=0.7, edgecolors="k", lw=0.5)
    ax.axhline(CONJUNC_KM, color="red", ls="--", lw=1, label=f"Conjunction threshold ({CONJUNC_KM} km)")
    ax.set_xlabel("Mean position 1-sigma (km)")
    ax.set_ylabel("Miss distance at TCA (km)")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No conjunction pairs", transform=ax.transAxes, ha="center")
ax.set_title("MC Dropout: Miss Distance vs Position Uncertainty")
ax.grid(True, alpha=0.3)

# Panel 6: Collision probability bar chart (log scale)
ax = axes[2, 1]
if conj_results:
    # Sort by Pc descending for readability
    sorted_res = sorted(conj_results, key=lambda r: r["Pc"], reverse=True)
    n_bars = min(20, len(sorted_res))  # show top 20
    labels = [f"A{r['iA']}-B{r['iB']}" for r in sorted_res[:n_bars]]
    pc_vals = [max(r["Pc"], 1e-30) for r in sorted_res[:n_bars]]  # floor for log
    colors = ["tab:red" if p > 1e-4 else "tab:blue" for p in pc_vals]
    y_pos = np.arange(n_bars)
    ax.barh(y_pos, pc_vals, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
    ax.axvline(1e-4, color="red", ls="--", lw=1.5, label="Operational threshold (1e-4)")
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Collision Probability (Pc)")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, "No conjunction pairs", transform=ax.transAxes, ha="center")
ax.set_title("Chan (2008) Collision Probability via MC Dropout")
ax.grid(True, alpha=0.3, axis="x")

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
    conj_label = f"Conjunctions (< {CONJUNC_KM} km)"
    f.write(f"{conj_label:<40}  {n_conjunctions:>18}\n")
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
    f.write("=" * 64 + "\n\n")

    # MC Dropout results
    f.write("MC Dropout Uncertainty Quantification\n")
    f.write(f"(n_mc={MC_SAMPLES}, dropout_p={DROPOUT_P}, r_combined={R_COMBINED*1000:.0f} m)\n")
    f.write("-" * 64 + "\n")
    f.write("NOTE: Model was trained WITHOUT dropout. These uncertainty\n")
    f.write("estimates are illustrative, not calibrated.\n\n")
    if conj_results:
        f.write(f"{'Pair':<12} {'Miss (km)':>10} {'sigma_x':>10} {'sigma_y':>10} {'sigma_z':>10} {'Pc':>12}\n")
        f.write("-" * 64 + "\n")
        for r in sorted(conj_results, key=lambda x: x["Pc"], reverse=True):
            f.write(f"A{r['iA']:<3}-B{r['iB']:<3}  "
                    f"{r['miss_km']:>10.3f} "
                    f"{r['sigma'][0]:>10.4f} "
                    f"{r['sigma'][1]:>10.4f} "
                    f"{r['sigma'][2]:>10.4f} "
                    f"{r['Pc']:>12.2e}\n")
    else:
        f.write("  No conjunction pairs found.\n")
    f.write("=" * 64 + "\n")
print(f"Saved: {table_path}")
print("\nDone.")
