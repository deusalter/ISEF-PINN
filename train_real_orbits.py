"""
Batch training on real satellite orbits -- ISEF PINN Project
=============================================================

For each satellite in the catalog, trains four models:
  1. Vanilla MLP    -- plain tanh NN, data-only
  2. Fourier NN     -- Fourier-featured NN, data-only
  3. Fourier PINN   -- Fourier-featured NN with J2-perturbed physics
  4. Fourier PINN   -- Fourier-featured NN with J2 + atmospheric drag physics

Data source:
  --data-source sgp4  (default): data/real_orbits/{norad_id}.npy
  --data-source gmat:            data/gmat_orbits/{norad_id}.npy

Results saved to:
  sgp4 -> data/real_results/{norad_id}_results.json
  gmat -> data/gmat_results/{norad_id}_results.json

Usage:
  python train_real_orbits.py                         # train all (SGP4 data)
  python train_real_orbits.py --sat 25544             # single satellite
  python train_real_orbits.py --data-source gmat      # train on GMAT data
"""

import sys
import os
import math
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import numpy as np

from src.physics import J2, J3, J4, R_EARTH, NormalizationParams
from src.atmosphere import drag_acceleration_torch
from src.models import FourierPINN, VanillaMLP, N_FREQ, HIDDEN, LAYERS
from satellite_catalog import get_catalog, get_by_norad_id

# -- Device -------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Hyperparameters ----------------------------------------------------------

TOTAL_EPOCHS   = 10000
LR_MAX         = 1e-3
LR_MIN         = 1e-5
SEC_LR_FACTOR  = 0.1     # secular head learns at 0.1 * LR_MAX (prevents overshoot)
N_COL          = 2000    # collocation points (doubled for better physics coverage)
TRAIN_FRAC     = 0.20    # first 20% = ~1 orbit
GRAD_CLIP      = 1.0

# 4-phase curriculum for SGP4 data (10K epochs).
# Uses full-domain collocation so physics guides test-region extrapolation.
# Lambda ramps to 0.15 -- aggressive enough to constrain extrapolation but
# moderate enough to tolerate small SGP4 vs J2+drag model mismatch.
CURRICULUM_SGP4 = [
    (2000,  0.00),   # Phase 1: data warmup only
    (5000,  0.01),   # Phase 2: gentle physics
    (8000,  0.05),   # Phase 3: moderate physics
    (10000, 0.15),   # Phase 4: full physics
]

# GMAT curriculum: 15K epochs with gentler physics ramp.
# GMAT dynamics are richer (full gravity, drag, 3rd body, SRP) so the J2-only
# physics loss should stay softer to avoid fighting unmodeled forces.
GMAT_EPOCHS = 15000
CURRICULUM_GMAT = [
    (3000,  0.00),   # Phase 1: data warmup
    (7000,  0.005),  # Phase 2: very gentle physics
    (11000, 0.02),   # Phase 3: moderate physics
    (15000, 0.05),   # Phase 4: full physics (capped at 0.05)
]


def get_lambda(ep, curriculum):
    """Return the physics weight for epoch ep based on curriculum schedule."""
    for end_ep, lam in curriculum:
        if ep <= end_ep:
            return lam
    return curriculum[-1][1]


def make_adaptive_collocation(t_norm_np, n_col, device):
    """Create adaptive collocation points concentrated near train/test boundary.

    Distribution:
      - 30% in training region (reinforce data fit)
      - 50% near train/test boundary (critical extrapolation transition)
      - 20% in far test region (physics guidance for long-range extrapolation)
    """
    t_min = float(t_norm_np[0]) + 0.001
    t_max = float(t_norm_np[-1])
    n_train_pts = int(len(t_norm_np) * TRAIN_FRAC)
    t_boundary = float(t_norm_np[n_train_pts - 1])

    # Boundary zone: from 80% of training region to 40% into test region
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

    t_col = torch.cat(parts).unsqueeze(1).requires_grad_(True)
    return t_col


# -- Physics loss (J2 in normalized coords, per-satellite R_NORM) -------------

def compute_j2_physics_loss(model, t_col, R_NORM, MU_NORM=1.0):
    """J2+J3 gravity physics residual in normalized coordinates.

    SGP4 includes J2 (dominant), J3 (odd harmonic, ~0.23% of J2), and J4
    (even harmonic, ~0.15% of J2).  Including J3 improves accuracy for
    inclined orbits where z != 0.  J4 is added as a small correction.
    All in normalized coordinates: pos_norm = pos_km / r_ref, t_norm = t_s / t_ref.
    """
    pos = model(t_col)
    ones = torch.ones(t_col.shape[0], dtype=torch.float64, device=t_col.device)

    # Velocity (first derivatives)
    vel = []
    for i in range(3):
        v_i = torch.autograd.grad(
            pos[:, i], t_col, ones,
            create_graph=True, retain_graph=True
        )[0]
        vel.append(v_i)
    vel = torch.cat(vel, dim=1)

    # Acceleration (second derivatives)
    acc = []
    for i in range(3):
        a_i = torch.autograd.grad(
            vel[:, i], t_col, ones,
            create_graph=True, retain_graph=True
        )[0]
        acc.append(a_i)
    acc = torch.cat(acc, dim=1)

    # Position components (normalized)
    x_n = pos[:, 0:1]
    y_n = pos[:, 1:2]
    z_n = pos[:, 2:3]

    r  = torch.norm(pos, dim=1, keepdim=True).clamp(min=1e-3)
    r2 = r ** 2
    r3 = r ** 3
    r5 = r ** 5
    r7 = r ** 7

    # Central gravity (scaled by MU_NORM)
    gravity = MU_NORM * pos / r3

    # -- J2 perturbation (dominant oblateness term) --
    z2_r2 = (z_n / r) ** 2
    j2_coeff = -1.5 * J2 * MU_NORM * (R_NORM ** 2) / r5
    a_j2_x = j2_coeff * x_n * (1.0 - 5.0 * z2_r2)
    a_j2_y = j2_coeff * y_n * (1.0 - 5.0 * z2_r2)
    a_j2_z = j2_coeff * z_n * (3.0 - 5.0 * z2_r2)
    a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

    # -- J3 perturbation (odd harmonic, north-south asymmetry, ~0.23% of J2) --
    # Derived from the gravitational potential:
    #   a_J3_x = -(5/2)*J3*mu*R_E^3/r^7 * x*(3z - 7z^3/r^2)
    #   a_J3_y = -(5/2)*J3*mu*R_E^3/r^7 * y*(3z - 7z^3/r^2)
    #   a_J3_z = -(J3*mu*R_E^3/2r^5)*(30z^2/r^2 - 35z^4/r^4 - 3)
    j3_xy_fac = -2.5 * J3 * MU_NORM * (R_NORM ** 3) / r7
    j3_xy_term = j3_xy_fac * (3.0 * z_n - 7.0 * z_n ** 3 / r2)
    a_j3_x = j3_xy_term * x_n
    a_j3_y = j3_xy_term * y_n
    a_j3_z = (-0.5 * J3 * MU_NORM * (R_NORM ** 3) / r5) * (
        30.0 * z_n ** 2 / r2 - 35.0 * z_n ** 4 / r2 ** 2 - 3.0
    )
    a_j3 = torch.cat([a_j3_x, a_j3_y, a_j3_z], dim=1)

    # -- J4 perturbation (even harmonic, ~0.15% of J2) --
    # a_J4_x = (15/8)*J4*mu*R_E^4/r^7 * x*(1 - 14z^2/r^2 + 21z^4/r^4)
    # a_J4_y = similar
    # a_J4_z = (5/8)*J4*mu*R_E^4/r^7 * z*(9 - 70z^2/r^2 + 63z^4/r^4)
    j4_fac = J4 * MU_NORM * (R_NORM ** 4) / r7
    z4_r4 = z_n ** 4 / r2 ** 2
    j4_xy = 1.875 * j4_fac * (1.0 - 14.0 * z2_r2 + 21.0 * z4_r4)
    a_j4_x = j4_xy * x_n
    a_j4_y = j4_xy * y_n
    a_j4_z = 0.625 * j4_fac * z_n * (9.0 - 70.0 * z2_r2 + 63.0 * z4_r4)
    a_j4 = torch.cat([a_j4_x, a_j4_y, a_j4_z], dim=1)

    # Residual: acc + gravity - a_J2 - a_J3 - a_J4 = 0
    residual = acc + gravity - a_j2 - a_j3 - a_j4

    return torch.mean(residual ** 2)


def compute_j2_drag_physics_loss(model, t_col, R_NORM, norm, MU_NORM, cd_a_over_m):
    """J2 + atmospheric drag physics residual in normalized coordinates.

    The model outputs normalized positions (pos_norm = pos_km / r_ref).
    Time is also normalized (t_norm = t_s / t_ref).

    Unit conversions applied here:
      pos_km    = pos_norm * r_ref
      vel_kms   = vel_norm * v_ref          (v_ref = r_ref / t_ref by definition)
      a_drag_normalized = a_drag_kms2 * (t_ref^2 / r_ref)

    Parameters
    ----------
    model         : FourierPINN
    t_col         : (N,1) tensor, normalized time, requires_grad=True
    R_NORM        : float, R_EARTH / a_km  (used in J2 term)
    norm          : NormalizationParams with .r_ref, .v_ref, .t_ref
    MU_NORM       : float, MU * t_ref^2 / r_ref^3  (close to 1.0 but not exact)
    cd_a_over_m   : float or scalar tensor, ballistic coeff Cd*A/m  [m^2/kg]
    """
    pos = model(t_col)
    ones = torch.ones(t_col.shape[0], dtype=torch.float64, device=t_col.device)

    # Velocity (first derivatives of normalized position w.r.t. normalized time)
    vel = []
    for i in range(3):
        v_i = torch.autograd.grad(
            pos[:, i], t_col, ones,
            create_graph=True, retain_graph=True
        )[0]
        vel.append(v_i)
    vel = torch.cat(vel, dim=1)

    # Acceleration (second derivatives)
    acc = []
    for i in range(3):
        a_i = torch.autograd.grad(
            vel[:, i], t_col, ones,
            create_graph=True, retain_graph=True
        )[0]
        acc.append(a_i)
    acc = torch.cat(acc, dim=1)

    # Two-body gravity (normalized)
    x_n = pos[:, 0:1]
    y_n = pos[:, 1:2]
    z_n = pos[:, 2:3]

    r = torch.norm(pos, dim=1, keepdim=True).clamp(min=1e-3)
    r3 = r ** 3
    r5 = r ** 5

    gravity = MU_NORM * pos / r3

    # Higher powers of r needed for J3/J4 terms
    r2 = r ** 2
    r7 = r5 * r2

    # -- J2 perturbation --
    z2_r2 = (z_n / r) ** 2
    j2_coeff = -1.5 * J2 * MU_NORM * (R_NORM ** 2) / r5
    a_j2_x = j2_coeff * x_n * (1.0 - 5.0 * z2_r2)
    a_j2_y = j2_coeff * y_n * (1.0 - 5.0 * z2_r2)
    a_j2_z = j2_coeff * z_n * (3.0 - 5.0 * z2_r2)
    a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

    # -- J3 perturbation (odd harmonic, ~0.23% of J2) --
    j3_xy_fac = -2.5 * J3 * MU_NORM * (R_NORM ** 3) / r7
    j3_xy_term = j3_xy_fac * (3.0 * z_n - 7.0 * z_n ** 3 / r2)
    a_j3_x = j3_xy_term * x_n
    a_j3_y = j3_xy_term * y_n
    a_j3_z = (-0.5 * J3 * MU_NORM * (R_NORM ** 3) / r5) * (
        30.0 * z_n ** 2 / r2 - 35.0 * z_n ** 4 / r2 ** 2 - 3.0
    )
    a_j3 = torch.cat([a_j3_x, a_j3_y, a_j3_z], dim=1)

    # -- J4 perturbation (even harmonic, ~0.15% of J2) --
    j4_fac = J4 * MU_NORM * (R_NORM ** 4) / r7
    z4_r4 = z_n ** 4 / r2 ** 2
    j4_xy = 1.875 * j4_fac * (1.0 - 14.0 * z2_r2 + 21.0 * z4_r4)
    a_j4_x = j4_xy * x_n
    a_j4_y = j4_xy * y_n
    a_j4_z = 0.625 * j4_fac * z_n * (9.0 - 70.0 * z2_r2 + 63.0 * z4_r4)
    a_j4 = torch.cat([a_j4_x, a_j4_y, a_j4_z], dim=1)

    # Atmospheric drag (convert to physical units, call drag model, renormalize)
    # pos_norm -> pos_km
    pos_km = pos * norm.r_ref
    # vel_norm (d pos_norm / d t_norm) -> vel_kms
    # vel_kms = d(pos_norm * r_ref) / d(t_norm * t_ref) = vel_norm * (r_ref / t_ref)
    # r_ref / t_ref == v_ref by definition
    vel_kms = vel * norm.v_ref

    # drag_acceleration_torch returns acceleration in km/s^2
    a_drag_kms2 = drag_acceleration_torch(pos_km, vel_kms, cd_a_over_m)

    # Convert drag back to normalized acceleration units:
    # a_norm = a_physical * (t_ref^2 / r_ref)
    a_drag_normalized = a_drag_kms2 * (norm.t_ref ** 2 / norm.r_ref)

    # Full residual: acc + gravity - a_j2 - a_j3 - a_j4 - a_drag = 0
    residual = acc + gravity - a_j2 - a_j3 - a_j4 - a_drag_normalized

    return torch.mean(residual ** 2)


# -- Training function (adapted from train_pinn_j2.py) ------------------------

def _make_optimizer(model, lr, use_separate_lr=False):
    """Create Adam optimizer, optionally with separate LR for secular head."""
    if use_separate_lr and hasattr(model, 'sec_head'):
        sec_params = list(model.sec_head.parameters())
        sec_ids = {id(p) for p in sec_params}
        main_params = [p for p in model.parameters() if id(p) not in sec_ids]
        param_groups = [
            {"params": main_params, "lr": lr},
            {"params": sec_params, "lr": lr * SEC_LR_FACTOR},
        ]
        return torch.optim.Adam(param_groups)
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_model(model, t_train, pos_train, t_col, t_all, pos_all_km,
                n_train, total_epochs, R_NORM, norm,
                use_physics=False, tag="Model",
                physics_loss_fn=None, curriculum=None):
    """Train a model using a phased curriculum schedule.

    Returns (best_test_rmse, best_train_rmse).

    Parameters
    ----------
    physics_loss_fn : callable or None
        If provided, called as physics_loss_fn(model, t_col).
        Must be passed explicitly when use_physics=True; there is no default.
    curriculum : list of (epoch, lambda) tuples or None
        Physics weight schedule. Defaults to CURRICULUM_SGP4.
    """
    if curriculum is None:
        curriculum = CURRICULUM_SGP4

    # Use separate LR for FourierPINN secular head
    use_sep_lr = hasattr(model, 'sec_head')
    optimizer = _make_optimizer(model, LR_MAX, use_separate_lr=use_sep_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=LR_MIN
    )
    mse = nn.MSELoss()

    best_test_rmse = float("inf")
    best_train_rmse = float("inf")
    best_state = None

    # Curriculum phase boundaries for optimizer refresh
    phase_boundaries = {ep for ep, _ in curriculum} if use_physics else set()

    model.train()
    for ep in range(1, total_epochs + 1):

        # Refresh Adam at each curriculum phase transition to reset stale momentum
        if use_physics and ep in phase_boundaries and ep > 1:
            optimizer = _make_optimizer(model, LR_MAX * 0.5, use_separate_lr=use_sep_lr)
            remaining = total_epochs - ep
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining, 1), eta_min=LR_MIN
            )

        optimizer.zero_grad()

        # Data loss
        pred = model(t_train)
        dl = mse(pred, pos_train)

        # Physics loss with curriculum lambda
        lam = get_lambda(ep, curriculum) if use_physics else 0.0
        if use_physics and lam > 0.0 and physics_loss_fn is not None:
            pl = physics_loss_fn(model, t_col)
            total = dl + lam * pl
            pv = pl.item()
        else:
            total = dl
            pv = 0.0

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # Best model tracking (every 500 epochs)
        if ep % 500 == 0 or ep == total_epochs:
            model.eval()
            with torch.no_grad():
                pred_km = model(t_all).cpu().numpy() * norm.r_ref
            err = np.linalg.norm(pred_km - pos_all_km, axis=1)
            test_rmse = float(np.sqrt(np.mean(err[n_train:] ** 2)))
            train_rmse = float(np.sqrt(np.mean(err[:n_train] ** 2)))
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_train_rmse = train_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

        # Print progress every 2500 epochs
        if ep % 2500 == 0 or ep == 1:
            lr = optimizer.param_groups[0]["lr"]
            phase_label = f"lam={lam:.3f}" if use_physics else "DATA"
            print(
                f"  [{tag:>6s}|{phase_label}] {ep:5d}/{total_epochs}  "
                f"d={dl.item():.4e}  p={pv:.4e}  "
                f"lr={lr:.2e}  best_test={best_test_rmse:.1f}km",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_test_rmse, best_train_rmse


# -- Main ---------------------------------------------------------------------

def train_satellite(norad_id, name, orbit_type, cd_a_over_m=0.022, verbose=True,
                    data_dir="data/real_orbits", results_dir="data/real_results",
                    data_source="sgp4"):
    """Train all four models for one satellite. Returns results dict or None."""

    data_path = f"{data_dir}/{norad_id}.npy"
    meta_path = f"{data_dir}/{norad_id}_meta.json"

    if not os.path.exists(data_path):
        print(f"  WARNING: data file not found: {data_path} -- skipping")
        return None
    if not os.path.exists(meta_path):
        print(f"  WARNING: meta file not found: {meta_path} -- skipping")
        return None

    # Load data and metadata
    data = np.load(data_path)  # (5000, 7): [t, x, y, z, vx, vy, vz]
    with open(meta_path, "r") as f:
        meta = json.load(f)

    a_km = meta["a_km"]
    N = data.shape[0]
    t_raw = data[:, 0]
    pos_km = data[:, 1:4]

    # Empirical t_ref: fit actual in-plane angular velocity from the SGP4 data.
    # This eliminates frequency mismatch that causes secular in-track drift,
    # analogous to what we do in train_pinn_j2.py for the synthetic orbit.
    _theta = np.unwrap(np.arctan2(data[:, 2], data[:, 1]))
    _omega  = np.polyfit(t_raw, _theta, 1)[0]    # rad/s
    _t_ref_emp = float(1.0 / _omega)             # s/rad -- empirical period/2pi

    # Per-satellite normalization using empirical t_ref
    norm   = NormalizationParams(r_ref=a_km, t_ref=_t_ref_emp)
    R_NORM = R_EARTH / a_km
    # MU_NORM < 1.0 when t_ref is J2-corrected (slightly shorter than Keplerian)
    MU_NORM = 398600.4418 * norm.t_ref ** 2 / norm.r_ref ** 3

    t_norm_np = t_raw / norm.t_ref
    pos_norm_np = pos_km / norm.r_ref

    n_train = int(N * TRAIN_FRAC)
    n_test = N - n_train

    if verbose:
        print(f"\n  Normalization: {norm}")
        print(f"  R_NORM = R_EARTH / a_km = {R_NORM:.6f}")
        print(f"  MU_NORM = {MU_NORM:.6f}")
        print(f"  Total: {N} pts | Train: {n_train} ({TRAIN_FRAC*100:.0f}%) | "
              f"Test: {n_test} ({(1-TRAIN_FRAC)*100:.0f}%)")
        print(f"  t_norm: [{t_norm_np[0]:.4f}, {t_norm_np[-1]:.4f}]")

    # Tensors (moved to GPU if available)
    t_train = torch.tensor(t_norm_np[:n_train, None], dtype=torch.float64).to(DEVICE)
    pos_train = torch.tensor(pos_norm_np[:n_train], dtype=torch.float64).to(DEVICE)
    t_all = torch.tensor(t_norm_np[:, None], dtype=torch.float64).to(DEVICE)

    # Select curriculum and epochs based on data source
    if data_source == "gmat":
        curriculum = CURRICULUM_GMAT
        total_epochs = GMAT_EPOCHS
    else:
        curriculum = CURRICULUM_SGP4
        total_epochs = TOTAL_EPOCHS

    # Adaptive collocation: concentrated near train/test boundary for better
    # extrapolation guidance. 30% train, 50% boundary, 20% far test region.
    t_col = make_adaptive_collocation(t_norm_np, N_COL, DEVICE)

    # -- Train Vanilla MLP ----------------------------------------------------
    if verbose:
        print(f"\n  {'-' * 60}")
        print(f"  Training Vanilla MLP (data-only, no Fourier, no physics)...")
        print(f"  {'-' * 60}")

    torch.manual_seed(42)
    vanilla = VanillaMLP().double().to(DEVICE)
    van_test_rmse, van_train_rmse = train_model(
        vanilla, t_train, pos_train, t_col, t_all, pos_km,
        n_train, total_epochs, R_NORM, norm,
        use_physics=False, tag="VAN",
        curriculum=curriculum,
    )

    # -- Train Fourier NN (data-only) -----------------------------------------
    if verbose:
        print(f"\n  {'-' * 60}")
        print(f"  Training Fourier NN (data-only, no physics)...")
        print(f"  {'-' * 60}")

    torch.manual_seed(42)
    fourier_nn = FourierPINN().double().to(DEVICE)
    nn_test_rmse, nn_train_rmse = train_model(
        fourier_nn, t_train, pos_train, t_col, t_all, pos_km,
        n_train, total_epochs, R_NORM, norm,
        use_physics=False, tag="F-NN",
        curriculum=curriculum,
    )

    # -- Train Fourier PINN (data + J2 physics) -------------------------------
    if verbose:
        print(f"\n  {'-' * 60}")
        print(f"  Training Fourier PINN (data + J2 physics, 4-phase curriculum)...")
        print(f"  {'-' * 60}")

    j2_loss_fn = lambda m, tc: compute_j2_physics_loss(m, tc, R_NORM, MU_NORM)
    torch.manual_seed(42)
    fourier_pinn = FourierPINN().double().to(DEVICE)
    pinn_test_rmse, pinn_train_rmse = train_model(
        fourier_pinn, t_train, pos_train, t_col, t_all, pos_km,
        n_train, total_epochs, R_NORM, norm,
        use_physics=True, tag="F-PINN",
        physics_loss_fn=j2_loss_fn,
        curriculum=curriculum,
    )

    # -- Train Fourier PINN (data + J2 + Drag physics, learnable Cd*A/m) -------
    if verbose:
        print(f"\n  {'-' * 60}")
        print(f"  Training Fourier PINN (J2+Drag, Cd*A/m=learnable "
              f"init={cd_a_over_m:.4f}, 4-phase curriculum)...")
        print(f"  {'-' * 60}")

    torch.manual_seed(42)
    fourier_pinn_drag = FourierPINN().double().to(DEVICE)
    # Make Cd*A/m a learnable scalar (log-space keeps it positive).
    # Gradient flows through the drag physics residual back to log_cd_a_over_m,
    # so the optimizer discovers the ballistic coefficient that best explains
    # the observed trajectory rather than relying on a guessed fixed value.
    fourier_pinn_drag.log_cd_a_over_m = nn.Parameter(
        torch.tensor(math.log(cd_a_over_m), dtype=torch.float64, device=DEVICE)
    )
    j2drag_loss_fn = lambda m, tc: compute_j2_drag_physics_loss(
        m, tc, R_NORM, norm, MU_NORM, torch.exp(m.log_cd_a_over_m)
    )
    drag_test_rmse, drag_train_rmse = train_model(
        fourier_pinn_drag, t_train, pos_train, t_col, t_all, pos_km,
        n_train, total_epochs, R_NORM, norm,
        use_physics=True, tag="J2+D",
        physics_loss_fn=j2drag_loss_fn,
        curriculum=curriculum,
    )
    learned_cd = float(torch.exp(fourier_pinn_drag.log_cd_a_over_m).detach())
    if verbose:
        print(f"  Learned Cd*A/m = {learned_cd:.4e}  (catalog init: {cd_a_over_m:.4e})")

    # Compute improvements
    pinn_improv = (
        (van_test_rmse - pinn_test_rmse) / van_test_rmse * 100
        if van_test_rmse > 0 else 0.0
    )
    drag_improv = (
        (van_test_rmse - drag_test_rmse) / van_test_rmse * 100
        if van_test_rmse > 0 else 0.0
    )

    # Build results dict
    results = {
        "norad_id": norad_id,
        "name": name,
        "orbit_type": orbit_type,
        "a_km": a_km,
        "inc_deg": meta.get("inc_deg", None),
        "ecc": meta.get("ecc", None),
        "period_s": meta.get("period_s", None),
        "cd_a_over_m_init": cd_a_over_m,
        "cd_a_over_m_learned": round(learned_cd, 6),
        "vanilla_test_rmse_km": round(van_test_rmse, 2),
        "fourier_nn_test_rmse_km": round(nn_test_rmse, 2),
        "fourier_pinn_test_rmse_km": round(pinn_test_rmse, 2),
        "j2drag_pinn_test_rmse_km": round(drag_test_rmse, 2),
        "vanilla_train_rmse_km": round(van_train_rmse, 2),
        "fourier_nn_train_rmse_km": round(nn_train_rmse, 2),
        "fourier_pinn_train_rmse_km": round(pinn_train_rmse, 2),
        "j2drag_pinn_train_rmse_km": round(drag_train_rmse, 2),
        "pinn_improvement_pct": round(pinn_improv, 2),
        "j2drag_improvement_pct": round(drag_improv, 2),
    }

    # Save per-satellite results
    os.makedirs(results_dir, exist_ok=True)
    result_path = f"{results_dir}/{norad_id}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n  Results saved -> {result_path}")
        print(f"  {'Model':<24s}  {'Train RMSE':>12s}  {'Test RMSE':>12s}")
        print(f"  {'-' * 24}  {'-' * 12}  {'-' * 12}")
        print(f"  {'Vanilla MLP':<24s}  {van_train_rmse:>10.2f} km  {van_test_rmse:>10.2f} km")
        print(f"  {'Fourier NN':<24s}  {nn_train_rmse:>10.2f} km  {nn_test_rmse:>10.2f} km")
        print(f"  {'Fourier PINN (J2)':<24s}  {pinn_train_rmse:>10.2f} km  {pinn_test_rmse:>10.2f} km")
        print(f"  {'Fourier PINN (J2+Drag)':<24s}  {drag_train_rmse:>10.2f} km  {drag_test_rmse:>10.2f} km  [Cd*A/m={learned_cd:.3e}]")
        print(f"  J2 PINN improvement over Vanilla:      {pinn_improv:+.1f}%")
        print(f"  J2+Drag PINN improvement over Vanilla:  {drag_improv:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train PINN models on real satellite orbits"
    )
    parser.add_argument(
        "--sat", type=int, default=None,
        help="NORAD ID of a single satellite to train (for testing)"
    )
    parser.add_argument(
        "--data-source", choices=["sgp4", "gmat"], default="sgp4",
        help="Data source: sgp4 (default) or gmat"
    )
    args = parser.parse_args()

    # Set directories based on data source
    if args.data_source == "gmat":
        data_dir = "data/gmat_orbits"
        results_dir = "data/gmat_results"
    else:
        data_dir = "data/real_orbits"
        results_dir = "data/real_results"

    print("=" * 70)
    print(f"  Real Satellite Orbits -- Fourier PINN Batch Training ({args.data_source.upper()} data)")
    print("=" * 70)

    device_str = (
        f"cuda ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "cpu"
    )
    print(f"\n  Device: {device_str}")

    _model_tmp = FourierPINN().double()
    n_params = sum(p.numel() for p in _model_tmp.parameters())
    n_sec = _model_tmp.sec_head.weight.numel()
    del _model_tmp

    if args.data_source == "gmat":
        curr = CURRICULUM_GMAT
        epochs = GMAT_EPOCHS
    else:
        curr = CURRICULUM_SGP4
        epochs = TOTAL_EPOCHS

    print(f"\n  FourierPINN: N_FREQ={N_FREQ}, {HIDDEN}x{LAYERS} tanh+residual + sec_head({n_sec}p), "
          f"{n_params:,} total params")
    print(f"  Collocation: {N_COL} adaptive points | 4-phase curriculum: {curr}")
    print(f"  Epochs: {epochs} | empirical t_ref per satellite")
    print(f"  Secular head LR: {SEC_LR_FACTOR}x main LR")
    print(f"  Train fraction: {TRAIN_FRAC*100:.0f}%")

    os.makedirs(results_dir, exist_ok=True)

    # Determine satellites to train
    if args.sat is not None:
        entry = get_by_norad_id(args.sat)
        if entry is None:
            print(f"\n  ERROR: NORAD ID {args.sat} not found in catalog.")
            sys.exit(1)
        satellites = [entry]
    else:
        satellites = get_catalog()

    print(f"\n  Satellites to train: {len(satellites)}")

    all_results = []
    for idx, sat in enumerate(satellites):
        print(f"\n{'=' * 70}")
        print(f"  [{idx+1}/{len(satellites)}] {sat.name} (NORAD {sat.norad_id}) "
              f"-- {sat.orbit_type}")
        print(f"  alt~{sat.approx_alt_km:.0f} km, inc~{sat.approx_inc_deg:.1f} deg, "
              f"ecc~{sat.approx_ecc:.4f}")
        print(f"{'=' * 70}")

        result = train_satellite(
            sat.norad_id, sat.name, sat.orbit_type,
            cd_a_over_m=sat.cd_a_over_m,
            data_dir=data_dir,
            results_dir=results_dir,
            data_source=args.data_source,
        )

        if result is not None:
            all_results.append(result)

    # -- Summary table --------------------------------------------------------
    if len(all_results) > 0:
        print(f"\n\n{'=' * 120}")
        print("  SUMMARY -- All Satellites")
        print(f"{'=' * 120}")
        print(
            f"  {'NORAD':>7s}  {'Name':<22s}  {'Type':<16s}  "
            f"{'Vanilla':>10s}  {'F-NN':>10s}  {'PINN(J2)':>10s}  "
            f"{'PINN(J2+D)':>12s}  {'Cd*A/m':>10s}  {'J2 Imp':>8s}  {'J2+D Imp':>8s}"
        )
        print(
            f"  {'-' * 7}  {'-' * 22}  {'-' * 16}  "
            f"{'-' * 10}  {'-' * 10}  {'-' * 10}  "
            f"{'-' * 12}  {'-' * 10}  {'-' * 8}  {'-' * 8}"
        )
        for r in all_results:
            cd_str = f"{r['cd_a_over_m_learned']:.3e}" if 'cd_a_over_m_learned' in r else "n/a"
            print(
                f"  {r['norad_id']:>7d}  {r['name']:<22s}  {r['orbit_type']:<16s}  "
                f"{r['vanilla_test_rmse_km']:>8.1f}km  "
                f"{r['fourier_nn_test_rmse_km']:>8.1f}km  "
                f"{r['fourier_pinn_test_rmse_km']:>8.1f}km  "
                f"{r['j2drag_pinn_test_rmse_km']:>10.1f}km  "
                f"{cd_str:>10s}  "
                f"{r['pinn_improvement_pct']:>+7.1f}%  "
                f"{r['j2drag_improvement_pct']:>+7.1f}%"
            )

        # Aggregate statistics
        van_avg = np.mean([r["vanilla_test_rmse_km"] for r in all_results])
        nn_avg = np.mean([r["fourier_nn_test_rmse_km"] for r in all_results])
        pinn_avg = np.mean([r["fourier_pinn_test_rmse_km"] for r in all_results])
        drag_avg = np.mean([r["j2drag_pinn_test_rmse_km"] for r in all_results])
        improv_avg = np.mean([r["pinn_improvement_pct"] for r in all_results])
        drag_improv_avg = np.mean([r["j2drag_improvement_pct"] for r in all_results])

        print(f"\n  {'AVERAGE':>7s}  {'':22s}  {'':16s}  "
              f"{van_avg:>8.1f}km  {nn_avg:>8.1f}km  {pinn_avg:>8.1f}km  "
              f"{drag_avg:>10.1f}km  "
              f"{improv_avg:>+7.1f}%  {drag_improv_avg:>+7.1f}%")

        # Save aggregate results
        aggregate_path = f"{results_dir}/summary.json"
        with open(aggregate_path, "w") as f:
            json.dump({
                "n_satellites": len(all_results),
                "results": all_results,
                "averages": {
                    "vanilla_test_rmse_km": round(float(van_avg), 2),
                    "fourier_nn_test_rmse_km": round(float(nn_avg), 2),
                    "fourier_pinn_test_rmse_km": round(float(pinn_avg), 2),
                    "j2drag_pinn_test_rmse_km": round(float(drag_avg), 2),
                    "pinn_improvement_pct": round(float(improv_avg), 2),
                    "j2drag_improvement_pct": round(float(drag_improv_avg), 2),
                },
            }, f, indent=2)
        print(f"\n  Aggregate results saved -> {aggregate_path}")

    else:
        print("\n  No satellites were successfully trained.")
        print(f"  Make sure data files exist in {data_dir}/")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
