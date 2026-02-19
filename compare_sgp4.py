"""
compare_sgp4.py
===============
Benchmark the Fourier-PINN against SGP4, the industry-standard orbit
propagator used by NORAD for satellite catalog maintenance and conjunction
screening.

SGP4 (Simplified General Perturbations 4) includes:
  - J2 oblateness perturbation
  - Atmospheric drag (BSTAR term)
  - Lunar / solar gravitational perturbations (long-period)

For our 400 km circular near-equatorial orbit with negligible drag (low
BSTAR), the dominant perturbation is J2.  Over 5 orbits (~7.7 hours) drag
and third-body effects are tiny, making this a clean J2-accuracy comparison.

Our ground truth is a high-accuracy DOP853 numerical integration of the
J2-perturbed ODE -- the same data on which the PINN was trained.

Outputs
-------
    figures/sgp4_comparison.png  -- error-vs-time log plot + bar chart
    figures/sgp4_table.txt       -- tabular accuracy summary
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.physics import MU, R_EARTH, R_ORBIT, NormalizationParams
from sgp4.api import Satrec, WGS72

# ── Directories ───────────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load ground truth and predictions ─────────────────────────────────────────
print("Loading data...")
data      = np.load("data/orbital_data_j2.npy")           # (5000, 7)
pinn_pred = np.load("data/pinn_j2_predictions.npy")        # (5000, 4)
van_pred  = np.load("data/vanilla_j2_predictions.npy")     # (5000, 4)

t_sec  = data[:, 0]      # seconds from epoch
gt_xyz = data[:, 1:4]    # km, ECI
n_pts  = len(t_sec)
N_TRAIN = int(n_pts * 0.20)

# ── Build SGP4 satellite ───────────────────────────────────────────────────────
print("Initialising SGP4 satellite...")
norm      = NormalizationParams(r_ref=R_ORBIT)
n_rad_s   = math.sqrt(MU / R_ORBIT ** 3)   # Keplerian mean motion [rad/s]
n_rad_min = n_rad_s * 60.0                 # [rad/min]
T_sec     = 2.0 * math.pi / n_rad_s       # orbital period [s]

# Epoch: Jan 1, 2024 00:00:00 UTC -- matches our simulation t=0
EPOCH_YYDDD = 24001.00000000   # YYDDD.DDDDDDDD
JD_EPOCH    = 2460310.5        # Julian date for 2024-01-01 00:00:00 UTC

satrec = Satrec()
satrec.sgp4init(
    WGS72,          # gravity model
    "i",            # mode: improved
    99999,          # satellite catalog number (synthetic)
    EPOCH_YYDDD,    # epoch
    1.0e-8,         # bstar (near-zero drag)
    0.0,            # ndot  (first deriv of mean motion -- zeroed)
    0.0,            # nddot (second deriv -- zeroed)
    1.0e-7,         # ecco  (near-zero eccentricity → circular orbit)
    0.0,            # argpo (argument of perigee [rad])
    0.0,            # inclo (inclination [rad]) -- equatorial
    0.0,            # mo    (mean anomaly [rad]) -- starts at x-axis
    n_rad_min,      # no_kozai (mean motion [rad/min])
    0.0,            # nodeo (RAAN [rad])
)

# ── Propagate SGP4 ────────────────────────────────────────────────────────────
print(f"Propagating {n_pts} SGP4 points...")
sgp4_xyz = np.zeros((n_pts, 3))
t0 = time.perf_counter()
for i, t in enumerate(t_sec):
    e, r, v = satrec.sgp4(JD_EPOCH, t / 86400.0)
    if e != 0:
        sgp4_xyz[i] = np.nan
    else:
        sgp4_xyz[i] = r
t_sgp4_ms = (time.perf_counter() - t0) * 1e3
print(f"  SGP4 propagation: {t_sgp4_ms:.1f} ms for {n_pts} points")

# ── Compute errors ────────────────────────────────────────────────────────────
err_pinn = np.linalg.norm(pinn_pred[:, 1:4] - gt_xyz, axis=1)
err_van  = np.linalg.norm(van_pred[:, 1:4]  - gt_xyz, axis=1)
err_sgp4 = np.linalg.norm(sgp4_xyz           - gt_xyz, axis=1)

mask_test = np.arange(n_pts) >= N_TRAIN

def test_rmse(err):
    return float(np.sqrt(np.mean(err[mask_test] ** 2)))

rmse_van  = test_rmse(err_van)
rmse_pinn = test_rmse(err_pinn)
rmse_sgp4 = test_rmse(err_sgp4)

print(f"\n== J2-Perturbed Test RMSE (last 80%) ==")
print(f"  Vanilla MLP         : {rmse_van:>10.2f} km")
print(f"  Fourier PINN (ours) : {rmse_pinn:>10.2f} km")
print(f"  SGP4                : {rmse_sgp4:>10.2f} km")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PINN vs SGP4: J2-Perturbed Orbital Propagation", fontsize=13, y=1.01)

# Left: instantaneous error over time (log scale)
ax = axes[0]
ax.plot(t_sec / 3600, err_van,  color="tab:red",   lw=0.7, alpha=0.8,
        label=f"Vanilla MLP ({rmse_van:.0f} km RMSE)")
ax.plot(t_sec / 3600, err_pinn, color="tab:blue",  lw=1.0,
        label=f"Fourier PINN (ours) ({rmse_pinn:.0f} km RMSE)")
ax.plot(t_sec / 3600, err_sgp4, color="tab:green", lw=1.0, linestyle="--",
        label=f"SGP4 ({rmse_sgp4:.1f} km RMSE)")
ax.axvline(t_sec[N_TRAIN] / 3600, color="k", linestyle=":", lw=1.2,
           label="Train / test split")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Position Error (km)")
ax.set_title("Instantaneous Position Error vs Time")
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: bar chart test RMSE
ax = axes[1]
labels = ["Vanilla\nMLP", "Fourier PINN\n(ours)", "SGP4"]
rmses  = [rmse_van, rmse_pinn, rmse_sgp4]
colors = ["tab:red", "tab:blue", "tab:green"]
bars   = ax.bar(labels, rmses, color=colors, alpha=0.85, edgecolor="k", lw=0.8)
for bar, v in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmses) * 0.02,
            f"{v:.0f} km", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Test RMSE (km)")
ax.set_title("Test-Set RMSE Comparison (5 orbits, 80% test)")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(rmses) * 1.25)

plt.tight_layout()
out_png = os.path.join(FIGURES_DIR, "sgp4_comparison.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Text table ────────────────────────────────────────────────────────────────
table_path = os.path.join(FIGURES_DIR, "sgp4_table.txt")
with open(table_path, "w") as f:
    f.write("J2-Perturbed Orbit -- SGP4 vs PINN vs Vanilla (Test RMSE)\n")
    f.write("=" * 55 + "\n")
    f.write(f"{'Model':<25}  {'Test RMSE (km)':>14}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Vanilla MLP':<25}  {rmse_van:>14.2f}\n")
    f.write(f"{'Fourier PINN (ours)':<25}  {rmse_pinn:>14.2f}\n")
    f.write(f"{'SGP4':<25}  {rmse_sgp4:>14.2f}\n")
    f.write("=" * 55 + "\n")
    f.write(f"\nNote: Ground truth = DOP853 numerical integration of J2 ODE.\n")
    f.write(f"SGP4 epoch: 2024-01-01 00:00:00 UTC, BSTAR = 1e-8 (near-zero drag).\n")
    f.write(f"Orbit: 400 km circular, equatorial, {T_sec/60:.1f} min period.\n")
print(f"Saved: {table_path}")
print("\nDone.")
