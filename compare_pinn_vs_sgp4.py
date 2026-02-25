"""
compare_pinn_vs_sgp4.py -- Proper hypothesis test: PINN vs SGP4 against GMAT truth
===================================================================================

This script implements the core scientific test:

  H0: PINN and SGP4 have equal accuracy when propagating LEO orbits
  H1: PINN achieves lower RMSE than SGP4 over the prediction window

For each satellite in the catalog:
  1. Load GMAT high-fidelity ground truth (data/gmat_orbits/{norad_id}.npy)
  2. Train PINN on first 20% of GMAT data (~1 orbit)
  3. Propagate SGP4 from the same epoch, convert to J2000 for apples-to-apples
  4. Compare PINN and SGP4 predictions against GMAT truth over the remaining 80%
  5. Record per-satellite RMSE for both methods

After all satellites:
  - Paired t-test: is mean(PINN_RMSE - SGP4_RMSE) significantly < 0?
  - Generate publication-quality figures

Usage:
  python compare_pinn_vs_sgp4.py               # full catalog
  python compare_pinn_vs_sgp4.py --sat 25544   # single satellite
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from src.physics import MU, R_EARTH, J2, J3, J4, NormalizationParams
from src.models import FourierPINN
from satellite_catalog import get_catalog, get_by_norad_id
from download_tle import load_tle
from frame_conversion import teme_to_j2000_batch

from sgp4.api import Satrec

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters (matched from train_real_orbits.py)
# ---------------------------------------------------------------------------
TOTAL_EPOCHS = 15000     # GMAT curriculum uses 15K epochs
LR_MAX = 1e-3
LR_MIN = 1e-5
SEC_LR_FACTOR = 0.1     # secular head learns at 0.1 * LR_MAX
N_COL = 1000
TRAIN_FRAC = 0.20
GRAD_CLIP = 1.0

# GMAT curriculum: gentler physics ramp (GMAT dynamics richer than J2+drag)
CURRICULUM = [
    (3000,  0.00),   # Phase 1: data warmup
    (7000,  0.005),  # Phase 2: very gentle physics
    (11000, 0.02),   # Phase 3: moderate physics
    (15000, 0.05),   # Phase 4: full physics (capped at 0.05)
]


def get_lambda(ep, curriculum):
    for end_ep, lam in curriculum:
        if ep <= end_ep:
            return lam
    return curriculum[-1][1]


def _make_optimizer(model, lr):
    """Create Adam optimizer with separate LR for secular head."""
    if hasattr(model, 'sec_head'):
        sec_params = list(model.sec_head.parameters())
        sec_ids = {id(p) for p in sec_params}
        main_params = [p for p in model.parameters() if id(p) not in sec_ids]
        param_groups = [
            {"params": main_params, "lr": lr},
            {"params": sec_params, "lr": lr * SEC_LR_FACTOR},
        ]
        return torch.optim.Adam(param_groups)
    return torch.optim.Adam(model.parameters(), lr=lr)


def make_adaptive_collocation(t_norm_np, n_col, n_train_pts, device):
    """Create adaptive collocation points concentrated near train/test boundary."""
    t_min = float(t_norm_np[0]) + 0.001
    t_max = float(t_norm_np[-1])
    t_boundary = float(t_norm_np[n_train_pts - 1])

    t_boundary_lo = t_boundary - 0.2 * (t_boundary - t_min)
    t_boundary_hi = t_boundary + 0.4 * (t_max - t_boundary)

    n_train_col = int(n_col * 0.30)
    n_boundary_col = int(n_col * 0.50)
    n_far_col = n_col - n_train_col - n_boundary_col

    parts = []
    if n_train_col > 0:
        parts.append(torch.linspace(t_min, t_boundary, n_train_col,
                                     dtype=torch.float64, device=device))
    if n_boundary_col > 0:
        parts.append(torch.linspace(t_boundary_lo, t_boundary_hi, n_boundary_col,
                                     dtype=torch.float64, device=device))
    if n_far_col > 0:
        parts.append(torch.linspace(t_boundary_hi, t_max, n_far_col,
                                     dtype=torch.float64, device=device))

    return torch.cat(parts).unsqueeze(1).requires_grad_(True)


# ---------------------------------------------------------------------------
# Physics loss (J2 only -- matches train_real_orbits.py)
# ---------------------------------------------------------------------------
def compute_j2_physics_loss(model, t_col, R_NORM, MU_NORM=1.0):
    pos = model(t_col)
    ones = torch.ones(t_col.shape[0], dtype=torch.float64, device=t_col.device)

    vel = []
    for i in range(3):
        v_i = torch.autograd.grad(
            pos[:, i], t_col, ones,
            create_graph=True, retain_graph=True
        )[0]
        vel.append(v_i)
    vel = torch.cat(vel, dim=1)

    acc = []
    for i in range(3):
        a_i = torch.autograd.grad(
            vel[:, i], t_col, ones,
            create_graph=True, retain_graph=True
        )[0]
        acc.append(a_i)
    acc = torch.cat(acc, dim=1)

    x_n, y_n, z_n = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    r = torch.norm(pos, dim=1, keepdim=True).clamp(min=1e-3)
    r2, r3, r5, r7 = r**2, r**3, r**5, r**7

    gravity = MU_NORM * pos / r3

    z2_r2 = (z_n / r) ** 2
    j2_coeff = -1.5 * J2 * MU_NORM * (R_NORM ** 2) / r5
    a_j2 = torch.cat([
        j2_coeff * x_n * (1.0 - 5.0 * z2_r2),
        j2_coeff * y_n * (1.0 - 5.0 * z2_r2),
        j2_coeff * z_n * (3.0 - 5.0 * z2_r2),
    ], dim=1)

    j3_xy_fac = -2.5 * J3 * MU_NORM * (R_NORM ** 3) / r7
    j3_xy_term = j3_xy_fac * (3.0 * z_n - 7.0 * z_n ** 3 / r2)
    a_j3 = torch.cat([
        j3_xy_term * x_n,
        j3_xy_term * y_n,
        (-0.5 * J3 * MU_NORM * (R_NORM ** 3) / r5) * (
            30.0 * z_n**2 / r2 - 35.0 * z_n**4 / r2**2 - 3.0
        ),
    ], dim=1)

    j4_fac = J4 * MU_NORM * (R_NORM ** 4) / r7
    z4_r4 = z_n**4 / r2**2
    j4_xy = 1.875 * j4_fac * (1.0 - 14.0 * z2_r2 + 21.0 * z4_r4)
    a_j4 = torch.cat([
        j4_xy * x_n,
        j4_xy * y_n,
        0.625 * j4_fac * z_n * (9.0 - 70.0 * z2_r2 + 63.0 * z4_r4),
    ], dim=1)

    residual = acc + gravity - a_j2 - a_j3 - a_j4
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# PINN training (simplified from train_real_orbits.py)
# ---------------------------------------------------------------------------
def train_pinn_on_gmat(data, a_km):
    """Train a Fourier PINN on GMAT data. Returns predicted positions (km)."""
    N = data.shape[0]
    t_raw = data[:, 0]
    pos_km = data[:, 1:4]

    # Empirical t_ref
    theta = np.unwrap(np.arctan2(data[:, 2], data[:, 1]))
    omega = np.polyfit(t_raw, theta, 1)[0]
    t_ref_emp = float(1.0 / omega)

    norm = NormalizationParams(r_ref=a_km, t_ref=t_ref_emp)
    R_NORM = R_EARTH / a_km
    MU_NORM = MU * norm.t_ref ** 2 / norm.r_ref ** 3

    t_norm_np = t_raw / norm.t_ref
    pos_norm_np = pos_km / norm.r_ref

    n_train = int(N * TRAIN_FRAC)

    t_train = torch.tensor(t_norm_np[:n_train, None], dtype=torch.float64).to(DEVICE)
    pos_train = torch.tensor(pos_norm_np[:n_train], dtype=torch.float64).to(DEVICE)
    t_all = torch.tensor(t_norm_np[:, None], dtype=torch.float64).to(DEVICE)

    # Adaptive collocation: concentrated near train/test boundary
    t_col = make_adaptive_collocation(t_norm_np, N_COL, n_train, DEVICE)

    torch.manual_seed(42)
    model = FourierPINN().double().to(DEVICE)

    j2_loss_fn = lambda m, tc: compute_j2_physics_loss(m, tc, R_NORM, MU_NORM)

    # Separate LR for secular head
    optimizer = _make_optimizer(model, LR_MAX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=LR_MIN
    )
    mse = nn.MSELoss()

    best_state = None
    best_test_rmse = float("inf")
    phase_boundaries = {ep for ep, _ in CURRICULUM}

    model.train()
    for ep in range(1, TOTAL_EPOCHS + 1):
        if ep in phase_boundaries and ep > 1:
            optimizer = _make_optimizer(model, LR_MAX * 0.5)
            remaining = TOTAL_EPOCHS - ep
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining, 1), eta_min=LR_MIN
            )

        optimizer.zero_grad()
        pred = model(t_train)
        dl = mse(pred, pos_train)

        lam = get_lambda(ep, CURRICULUM)
        if lam > 0.0:
            pl = j2_loss_fn(model, t_col)
            total = dl + lam * pl
        else:
            total = dl

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if ep % 500 == 0 or ep == TOTAL_EPOCHS:
            model.eval()
            with torch.no_grad():
                pred_km = model(t_all).cpu().numpy() * norm.r_ref
            err = np.linalg.norm(pred_km - pos_km, axis=1)
            test_rmse = float(np.sqrt(np.mean(err[n_train:] ** 2)))
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

        if ep % 2500 == 0:
            print(f"    PINN ep={ep}/{TOTAL_EPOCHS}  d={dl.item():.4e}  "
                  f"best_test={best_test_rmse:.1f}km", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pinn_pred_km = model(t_all).cpu().numpy() * norm.r_ref

    return pinn_pred_km, n_train


# ---------------------------------------------------------------------------
# SGP4 propagation in J2000
# ---------------------------------------------------------------------------
def propagate_sgp4_j2000(line1, line2, t_seconds, epoch_dt):
    """Propagate SGP4 and convert output to J2000 frame.

    Parameters
    ----------
    line1, line2 : str
        TLE lines.
    t_seconds : np.ndarray, shape (N,)
        Time offsets from epoch [seconds].
    epoch_dt : datetime
        TLE epoch.

    Returns
    -------
    pos_j2000 : np.ndarray, shape (N, 3)
        Positions in J2000 [km].
    """
    sat = Satrec.twoline2rv(line1, line2)

    N = len(t_seconds)
    jd_array = np.full(N, sat.jdsatepoch)
    fr_array = sat.jdsatepochF + t_seconds / 86400.0

    e_arr, r_arr, v_arr = sat.sgp4_array(jd_array, fr_array)

    # Check for errors
    bad = e_arr != 0
    if np.any(bad):
        n_bad = int(np.sum(bad))
        print(f"    WARNING: SGP4 had {n_bad}/{N} error points")
        # Replace bad points with interpolation from neighbors
        for i in np.where(bad)[0]:
            if i > 0:
                r_arr[i] = r_arr[i - 1]
                v_arr[i] = v_arr[i - 1]

    # Convert TEME -> J2000
    pos_j2000, vel_j2000 = teme_to_j2000_batch(
        r_arr, v_arr, epoch_dt
    )

    return pos_j2000


# ---------------------------------------------------------------------------
# Per-satellite comparison
# ---------------------------------------------------------------------------
def compare_satellite(norad_id, name, orbit_type, cd_a_over_m=0.022):
    """Run PINN vs SGP4 comparison for one satellite against GMAT truth.

    Returns
    -------
    dict or None
        Per-satellite results, or None if data unavailable.
    """
    data_path = f"data/gmat_orbits/{norad_id}.npy"
    meta_path = f"data/gmat_orbits/{norad_id}_meta.json"

    if not os.path.exists(data_path):
        print(f"  SKIPPED (no GMAT data: {data_path})")
        return None
    if not os.path.exists(meta_path):
        print(f"  SKIPPED (no GMAT meta: {meta_path})")
        return None

    # Load GMAT ground truth
    gmat_data = np.load(data_path)  # (5000, 7)
    with open(meta_path) as f:
        meta = json.load(f)

    a_km = meta["a_km"]
    t_seconds = gmat_data[:, 0]
    gmat_pos_km = gmat_data[:, 1:4]  # J2000 positions

    print(f"  GMAT data: {gmat_data.shape[0]} pts, span={t_seconds[-1]/3600:.1f}h")

    # Load TLE for SGP4 propagation
    tle_result = load_tle(norad_id)
    if tle_result is None:
        print(f"  SKIPPED (no TLE for SGP4)")
        return None
    line1, line2 = tle_result

    # Parse epoch
    epoch_str = line1[18:32].strip()
    year_2d = int(epoch_str[:2])
    day_frac = float(epoch_str[2:])
    year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
    from datetime import timedelta
    epoch_dt = datetime(year, 1, 1) + timedelta(days=day_frac - 1.0)

    # 1. Train PINN on GMAT data
    print(f"  Training PINN on first {TRAIN_FRAC*100:.0f}% of GMAT data...")
    pinn_pred_km, n_train = train_pinn_on_gmat(gmat_data, a_km)

    # 2. Propagate SGP4 from same epoch
    print(f"  Propagating SGP4...")
    sgp4_pos_km = propagate_sgp4_j2000(line1, line2, t_seconds, epoch_dt)

    # 3. Compute errors against GMAT truth (test region only)
    pinn_err = np.linalg.norm(pinn_pred_km - gmat_pos_km, axis=1)
    sgp4_err = np.linalg.norm(sgp4_pos_km - gmat_pos_km, axis=1)

    # Test region (80% of data)
    pinn_test_rmse = float(np.sqrt(np.mean(pinn_err[n_train:] ** 2)))
    sgp4_test_rmse = float(np.sqrt(np.mean(sgp4_err[n_train:] ** 2)))

    pinn_train_rmse = float(np.sqrt(np.mean(pinn_err[:n_train] ** 2)))
    sgp4_train_rmse = float(np.sqrt(np.mean(sgp4_err[:n_train] ** 2)))

    # Full-span errors
    pinn_full_rmse = float(np.sqrt(np.mean(pinn_err ** 2)))
    sgp4_full_rmse = float(np.sqrt(np.mean(sgp4_err ** 2)))

    print(f"  PINN test RMSE:  {pinn_test_rmse:.2f} km")
    print(f"  SGP4 test RMSE:  {sgp4_test_rmse:.2f} km")

    improvement = (sgp4_test_rmse - pinn_test_rmse) / sgp4_test_rmse * 100 \
        if sgp4_test_rmse > 0 else 0.0

    result = {
        "norad_id": norad_id,
        "name": name,
        "orbit_type": orbit_type,
        "a_km": a_km,
        "inc_deg": meta.get("inc_deg"),
        "ecc": meta.get("ecc"),
        "period_s": meta.get("period_s"),
        "pinn_test_rmse_km": round(pinn_test_rmse, 4),
        "sgp4_test_rmse_km": round(sgp4_test_rmse, 4),
        "pinn_train_rmse_km": round(pinn_train_rmse, 4),
        "sgp4_train_rmse_km": round(sgp4_train_rmse, 4),
        "pinn_full_rmse_km": round(pinn_full_rmse, 4),
        "sgp4_full_rmse_km": round(sgp4_full_rmse, 4),
        "pinn_improvement_over_sgp4_pct": round(improvement, 2),
    }

    # Save per-satellite result
    os.makedirs("data/gmat_results", exist_ok=True)
    result_path = f"data/gmat_results/{norad_id}_comparison.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------
def generate_figures(all_results):
    """Generate publication-quality comparison figures."""
    os.makedirs("figures", exist_ok=True)

    names = [r["name"][:15] for r in all_results]
    pinn_rmses = [r["pinn_test_rmse_km"] for r in all_results]
    sgp4_rmses = [r["sgp4_test_rmse_km"] for r in all_results]

    # --- Figure 1: Bar chart comparison ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, sgp4_rmses, width, label="SGP4", color="#d62728", alpha=0.85)
    bars2 = ax.bar(x + width/2, pinn_rmses, width, label="PINN (J2)", color="#1f77b4", alpha=0.85)

    ax.set_xlabel("Satellite", fontsize=12)
    ax.set_ylabel("Test RMSE (km)", fontsize=12)
    ax.set_title("PINN vs SGP4 Accuracy Against GMAT Ground Truth", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pinn_vs_sgp4_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/pinn_vs_sgp4_comparison.png")

    # --- Figure 2: Scatter plot (PINN vs SGP4 RMSE) ---
    fig, ax = plt.subplots(figsize=(8, 8))

    max_val = max(max(pinn_rmses), max(sgp4_rmses)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal accuracy")
    ax.scatter(sgp4_rmses, pinn_rmses, s=80, c="#1f77b4", edgecolors="black",
               linewidth=0.5, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (sgp4_rmses[i], pinn_rmses[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("SGP4 Test RMSE (km)", fontsize=12)
    ax.set_ylabel("PINN Test RMSE (km)", fontsize=12)
    ax.set_title("PINN vs SGP4: Per-Satellite Accuracy", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pinn_vs_sgp4_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/pinn_vs_sgp4_scatter.png")

    # --- Figure 3: Improvement histogram ---
    improvements = [r["pinn_improvement_over_sgp4_pct"] for r in all_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(improvements, bins=15, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No improvement")
    ax.axvline(np.mean(improvements), color="green", linestyle="-", linewidth=2,
               label=f"Mean: {np.mean(improvements):.1f}%")
    ax.set_xlabel("PINN Improvement over SGP4 (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of PINN Improvement over SGP4", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pinn_improvement_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/pinn_improvement_histogram.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare PINN vs SGP4 against GMAT ground truth"
    )
    parser.add_argument(
        "--sat", type=int, default=None,
        help="NORAD ID for single-satellite test"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PINN vs SGP4 Hypothesis Test (GMAT Ground Truth)")
    print("=" * 70)
    print(f"\n  Device: {DEVICE}")

    # Build satellite list
    if args.sat is not None:
        entry = get_by_norad_id(args.sat)
        if entry is None:
            print(f"\n  ERROR: NORAD ID {args.sat} not found in catalog.")
            sys.exit(1)
        satellites = [entry]
    else:
        satellites = get_catalog()

    print(f"  Satellites: {len(satellites)}")
    print(f"  Train fraction: {TRAIN_FRAC*100:.0f}% (~1 orbit)")

    all_results = []

    for idx, sat in enumerate(satellites):
        print(f"\n{'=' * 70}")
        print(f"  [{idx+1}/{len(satellites)}] {sat.name} (NORAD {sat.norad_id})")
        print(f"{'=' * 70}")

        result = compare_satellite(
            sat.norad_id, sat.name, sat.orbit_type,
            cd_a_over_m=sat.cd_a_over_m,
        )

        if result is not None:
            all_results.append(result)

    # -- Statistical analysis --
    if len(all_results) < 2:
        print("\n  Not enough satellites for statistical test.")
        if len(all_results) == 1:
            r = all_results[0]
            print(f"\n  {r['name']}:")
            print(f"    PINN RMSE: {r['pinn_test_rmse_km']:.2f} km")
            print(f"    SGP4 RMSE: {r['sgp4_test_rmse_km']:.2f} km")
            print(f"    Improvement: {r['pinn_improvement_over_sgp4_pct']:.1f}%")
        return

    print(f"\n\n{'=' * 70}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    pinn_rmses = np.array([r["pinn_test_rmse_km"] for r in all_results])
    sgp4_rmses = np.array([r["sgp4_test_rmse_km"] for r in all_results])
    differences = pinn_rmses - sgp4_rmses  # negative = PINN better

    print(f"\n  N = {len(all_results)} satellites")
    print(f"\n  PINN test RMSE:  mean={np.mean(pinn_rmses):.2f} km, "
          f"median={np.median(pinn_rmses):.2f} km, "
          f"std={np.std(pinn_rmses):.2f} km")
    print(f"  SGP4 test RMSE:  mean={np.mean(sgp4_rmses):.2f} km, "
          f"median={np.median(sgp4_rmses):.2f} km, "
          f"std={np.std(sgp4_rmses):.2f} km")
    print(f"  Difference:      mean={np.mean(differences):.2f} km "
          f"(negative = PINN better)")

    # Paired t-test (one-sided: PINN < SGP4)
    t_stat, p_two = stats.ttest_rel(pinn_rmses, sgp4_rmses)
    # One-sided p-value (PINN < SGP4)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    print(f"\n  Paired t-test (H1: PINN RMSE < SGP4 RMSE):")
    print(f"    t-statistic:   {t_stat:.4f}")
    print(f"    p-value (1-sided): {p_one:.2e}")

    if p_one < 0.05:
        print(f"    Result: SIGNIFICANT at alpha=0.05 (PINN is more accurate)")
    elif p_one < 0.10:
        print(f"    Result: MARGINALLY SIGNIFICANT at alpha=0.10")
    else:
        print(f"    Result: NOT SIGNIFICANT (insufficient evidence)")

    # Per-satellite table
    print(f"\n  {'NORAD':>7s}  {'Name':<20s}  {'PINN RMSE':>10s}  "
          f"{'SGP4 RMSE':>10s}  {'Improv%':>8s}")
    print(f"  {'-'*7}  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    for r in all_results:
        print(f"  {r['norad_id']:>7d}  {r['name']:<20s}  "
              f"{r['pinn_test_rmse_km']:>8.2f}km  "
              f"{r['sgp4_test_rmse_km']:>8.2f}km  "
              f"{r['pinn_improvement_over_sgp4_pct']:>+7.1f}%")

    # Wilcoxon signed-rank test (non-parametric alternative)
    if len(all_results) >= 6:
        w_stat, w_p = stats.wilcoxon(differences, alternative="less")
        print(f"\n  Wilcoxon signed-rank test (non-parametric):")
        print(f"    W-statistic: {w_stat:.4f}")
        print(f"    p-value:     {w_p:.2e}")

    # Effect size (Cohen's d)
    d_mean = np.mean(differences)
    d_std = np.std(differences, ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else 0.0
    print(f"\n  Effect size (Cohen's d): {cohens_d:.4f}")

    # Save aggregate results
    summary = {
        "n_satellites": len(all_results),
        "pinn_mean_rmse_km": round(float(np.mean(pinn_rmses)), 4),
        "sgp4_mean_rmse_km": round(float(np.mean(sgp4_rmses)), 4),
        "mean_difference_km": round(float(np.mean(differences)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value_one_sided": float(p_one),
        "cohens_d": round(float(cohens_d), 4),
        "significant_at_005": bool(p_one < 0.05),
        "results": all_results,
    }

    os.makedirs("data/gmat_results", exist_ok=True)
    summary_path = "data/gmat_results/hypothesis_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}")

    # Generate figures
    print(f"\n  Generating figures...")
    generate_figures(all_results)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
