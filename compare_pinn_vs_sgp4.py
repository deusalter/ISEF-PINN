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
  python compare_pinn_vs_sgp4.py --long-arc    # use 7-day GMAT data
"""

import argparse
import json
import math
import os
import sys
import time
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

from src.physics import MU, R_EARTH, J2, J3, J4, J5, NormalizationParams
from src.models import FourierPINN, NeuralODE
from src.atmosphere import drag_acceleration_torch
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
        0.625 * j4_fac * z_n * (15.0 - 70.0 * z2_r2 + 63.0 * z4_r4),
    ], dim=1)

    # -- J5 perturbation (odd harmonic, ~0.02% of J2) --
    r9 = r7 * r2
    z6_r6 = z4_r4 * z2_r2
    j5_fac = J5 * MU_NORM * (R_NORM ** 5)
    j5_xy = (21.0 / 8.0) * j5_fac * z_n / r9 * (33.0 * z4_r4 - 30.0 * z2_r2 + 5.0)
    a_j5 = torch.cat([
        j5_xy * x_n,
        j5_xy * y_n,
        (3.0 / 8.0) * j5_fac / r7 * (231.0 * z6_r6 - 315.0 * z4_r4 + 105.0 * z2_r2 - 5.0),
    ], dim=1)

    residual = acc + gravity - a_j2 - a_j3 - a_j4 - a_j5
    return torch.mean(residual ** 2)


def compute_j2_drag_physics_loss(model, t_col, R_NORM, norm, MU_NORM, cd_a_over_m):
    """J2-J5 + atmospheric drag physics residual (for long-arc propagation)."""
    pos = model(t_col)
    ones = torch.ones(t_col.shape[0], dtype=torch.float64, device=t_col.device)

    vel = []
    for i in range(3):
        v_i = torch.autograd.grad(
            pos[:, i], t_col, ones, create_graph=True, retain_graph=True
        )[0]
        vel.append(v_i)
    vel = torch.cat(vel, dim=1)

    acc = []
    for i in range(3):
        a_i = torch.autograd.grad(
            vel[:, i], t_col, ones, create_graph=True, retain_graph=True
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
        0.625 * j4_fac * z_n * (15.0 - 70.0 * z2_r2 + 63.0 * z4_r4),
    ], dim=1)

    r9 = r7 * r2
    z6_r6 = z4_r4 * z2_r2
    j5_fac = J5 * MU_NORM * (R_NORM ** 5)
    j5_xy = (21.0 / 8.0) * j5_fac * z_n / r9 * (33.0 * z4_r4 - 30.0 * z2_r2 + 5.0)
    a_j5 = torch.cat([
        j5_xy * x_n,
        j5_xy * y_n,
        (3.0 / 8.0) * j5_fac / r7 * (231.0 * z6_r6 - 315.0 * z4_r4 + 105.0 * z2_r2 - 5.0),
    ], dim=1)

    # Atmospheric drag (convert to physical units, call drag model, renormalize)
    pos_km = pos * norm.r_ref
    vel_kms = vel * norm.v_ref
    a_drag_kms2 = drag_acceleration_torch(pos_km, vel_kms, cd_a_over_m)
    a_drag_normalized = a_drag_kms2 * (norm.t_ref ** 2 / norm.r_ref)

    residual = acc + gravity - a_j2 - a_j3 - a_j4 - a_j5 - a_drag_normalized
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# PINN training (simplified from train_real_orbits.py)
# ---------------------------------------------------------------------------
def train_pinn_on_gmat(data, a_km):
    """Train a Fourier PINN on GMAT data. Returns predicted positions (km).

    Parameters
    ----------
    data : np.ndarray, shape (N, 7)
        GMAT data: [t_seconds, x, y, z, vx, vy, vz].
        Time column should start near 0 (rebased if using a late window).
    a_km : float
        Semi-major axis for normalization.

    Returns
    -------
    pinn_pred_km : np.ndarray, shape (N, 3)
    n_train : int
    model, t_all, norm : for inference timing
    """
    N = data.shape[0]
    t_raw = data[:, 0]
    pos_km = data[:, 1:4]

    # Empirical t_ref from orbital angular velocity
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

    t_col = make_adaptive_collocation(t_norm_np, N_COL, n_train, DEVICE)

    torch.manual_seed(42)
    model = FourierPINN().double().to(DEVICE)

    physics_loss_fn = lambda m, tc: compute_j2_physics_loss(m, tc, R_NORM, MU_NORM)

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
            pl = physics_loss_fn(model, t_col)
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

    return pinn_pred_km, n_train, model, t_all, norm


# ---------------------------------------------------------------------------
# Neural ODE training (for --long-arc mode)
# ---------------------------------------------------------------------------
NODE_EPOCHS = 2000       # ~35 min per satellite on CPU
NODE_LR = 3e-4           # Adam, cosine anneal to 1e-6
NODE_LR_MIN = 1e-6
NODE_DT = 60.0           # RK4 step size (seconds)
NODE_N_SEGMENTS = 20     # multiple shooting segments in training window
NODE_VEL_WEIGHT = 0.1    # velocity loss weight (position = 1.0)
NODE_HIDDEN = 32
NODE_LAYERS = 2


def train_neural_ode_on_gmat(data, a_km, cd_a_over_m):
    """Train a Neural ODE on GMAT data for long-arc propagation.

    Uses multiple-shooting: splits training window into segments, integrates
    each from a known GMAT state, and trains the NN correction to minimize
    position + velocity error.

    Parameters
    ----------
    data : np.ndarray, shape (N, 7)
        GMAT data: [t_seconds, x, y, z, vx, vy, vz].
    a_km : float
        Semi-major axis for normalization references.
    cd_a_over_m : float
        Ballistic coefficient (unused by Neural ODE directly, but kept for API).

    Returns
    -------
    ode_pred : np.ndarray, shape (N, 6)
        Predicted [x, y, z, vx, vy, vz] in km and km/s.
    n_train : int
        Number of training points.
    model : NeuralODE
        Trained model.
    norm : NormalizationParams
        Normalization parameters used.
    """
    N = data.shape[0]
    t_seconds = data[:, 0]
    states_km = data[:, 1:7]  # (N, 6) [x, y, z, vx, vy, vz]

    # Normalization references
    from src.physics import circular_velocity
    v_ref = circular_velocity(a_km)
    norm = NormalizationParams(r_ref=a_km, v_ref=v_ref)

    # Train/test split: 20% train, 80% test
    n_train = int(N * TRAIN_FRAC)

    # Build model
    torch.manual_seed(42)
    model = NeuralODE(
        r_ref=a_km, v_ref=v_ref,
        hidden=NODE_HIDDEN, n_layers=NODE_LAYERS,
    ).double().to(DEVICE)

    n_params = sum(p.numel() for p in model.correction.parameters())
    print(f"    Neural ODE: {n_params} NN parameters, "
          f"dt={NODE_DT}s, {NODE_N_SEGMENTS} segments")

    optimizer = torch.optim.Adam(model.parameters(), lr=NODE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NODE_EPOCHS, eta_min=NODE_LR_MIN
    )

    # Prepare training segments (multiple shooting)
    train_t = t_seconds[:n_train]
    train_states = states_km[:n_train]  # (n_train, 6)
    pts_per_seg = max(1, n_train // NODE_N_SEGMENTS)
    seg_starts = list(range(0, n_train, pts_per_seg))
    if not seg_starts:
        seg_starts = [0]
    n_segs = len(seg_starts)
    print(f"    Training: {n_train} pts, {n_segs} segments, "
          f"~{pts_per_seg} pts/seg")

    # Pre-convert segment data to tensors
    seg_data = []
    for si in seg_starts:
        se = min(si + pts_per_seg, n_train)
        s0 = torch.tensor(train_states[si], dtype=torch.float64, device=DEVICE)
        t_seg = torch.tensor(train_t[si:se], dtype=torch.float64, device=DEVICE)
        truth_seg = torch.tensor(
            train_states[si:se], dtype=torch.float64, device=DEVICE
        )
        seg_data.append((s0, t_seg, truth_seg))

    # Prepare batched tensors for fast vectorized training
    all_s0 = torch.stack([s0 for s0, _, _ in seg_data], dim=0)  # (n_segs, 6)
    ref_t_seg = seg_data[0][1]
    rel_times = ref_t_seg - ref_t_seg[0]  # relative times (starts at 0)
    min_seg_pts = min(truth.shape[0] for _, _, truth in seg_data)
    rel_times = rel_times[:min_seg_pts]
    all_truth = torch.stack(
        [truth[:min_seg_pts] for _, _, truth in seg_data], dim=0
    )  # (n_segs, min_seg_pts, 6)

    best_state = None
    best_loss = float("inf")

    model.train()
    for ep in range(1, NODE_EPOCHS + 1):
        optimizer.zero_grad()

        # Batched forward: all segments integrated simultaneously
        pred_batch = model.integrate_batched(
            all_s0, rel_times, dt=NODE_DT
        )  # (n_segs, min_seg_pts, 6)

        # Position loss (km)
        pos_loss = torch.mean((pred_batch[:, :, :3] - all_truth[:, :, :3]) ** 2)
        # Velocity loss (km/s)
        vel_loss = torch.mean((pred_batch[:, :, 3:] - all_truth[:, :, 3:]) ** 2)
        total_loss = pos_loss + NODE_VEL_WEIGHT * vel_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # Checkpoint best model
        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 200 == 0 or ep == NODE_EPOCHS:
            lr_now = scheduler.get_last_lr()[0]
            log_s = model.correction.log_scale.item()
            print(f"    NODE ep={ep}/{NODE_EPOCHS}  loss={loss_val:.4e}  "
                  f"best={best_loss:.4e}  lr={lr_now:.1e}  "
                  f"log_scale={log_s:.2f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test prediction: single-shot from training boundary ---
    model.eval()
    print(f"    Generating predictions (train + test)...")

    with torch.no_grad():
        # Train region: batched segment integration
        train_pred_batch = model.integrate_batched(
            all_s0, rel_times, dt=NODE_DT
        )  # (n_segs, min_seg_pts, 6)
        train_pred = train_pred_batch.reshape(-1, 6)  # (~n_train, 6)

        # Test region: single-shot integration from last training state
        t0_test = t_seconds[n_train - 1]
        state0_test = torch.tensor(
            states_km[n_train - 1], dtype=torch.float64, device=DEVICE
        )
        t_test = torch.tensor(
            t_seconds[n_train:], dtype=torch.float64, device=DEVICE
        )
        test_pred = model.integrate_to_times(
            state0_test, t0_test, t_test, dt=NODE_DT
        )  # (N - n_train, 6)

    # Combine train + test predictions
    all_pred = torch.cat([train_pred, test_pred], dim=0)
    ode_pred = all_pred.cpu().numpy()

    # Pad if segment-based train prediction has fewer points than n_train
    if ode_pred.shape[0] < N:
        pad = np.tile(ode_pred[-1:], (N - ode_pred.shape[0], 1))
        ode_pred = np.concatenate([ode_pred, pad], axis=0)
    # Trim if we have excess from segment boundaries
    ode_pred = ode_pred[:N]

    return ode_pred, n_train, model, norm


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
def compare_satellite(norad_id, name, orbit_type, cd_a_over_m=0.022,
                      long_arc=False):
    """Run PINN/NeuralODE vs SGP4 comparison for one satellite against GMAT truth.

    For long_arc mode, uses Neural ODE:
      - Load full 7-day GMAT data (no windowing)
      - Train Neural ODE on first 20% (multiple shooting with J2-J5 + learned corrections)
      - SGP4 propagates from TLE epoch
      - Compare both on the remaining 80% test window
    This is a fair comparison: both methods evaluated on the same test window.

    For 5-orbit mode, uses FourierPINN (unchanged).

    Returns
    -------
    dict or None
        Per-satellite results, or None if data unavailable.
    """
    suffix = "_7day" if long_arc else ""
    data_path = f"data/gmat_orbits/{norad_id}{suffix}.npy"
    meta_path = f"data/gmat_orbits/{norad_id}_meta.json"

    if not os.path.exists(data_path):
        print(f"  SKIPPED (no GMAT data: {data_path})")
        return None
    if not os.path.exists(meta_path):
        print(f"  SKIPPED (no GMAT meta: {meta_path})")
        return None

    # Load GMAT ground truth
    gmat_data_full = np.load(data_path)
    with open(meta_path) as f:
        meta = json.load(f)

    a_km = meta["a_km"]

    if long_arc:
        # Use ALL 7-day data — no late-window trick
        gmat_data = gmat_data_full
        t_seconds = gmat_data[:, 0]
        t_seconds_orig = t_seconds
        gmat_pos_km = gmat_data[:, 1:4]
        print(f"  7-day data: {gmat_data.shape[0]} pts, "
              f"span={t_seconds[-1]/3600:.1f}h ({t_seconds[-1]/86400:.1f} days)")
    else:
        gmat_data = gmat_data_full
        t_seconds = gmat_data[:, 0]
        t_seconds_orig = t_seconds  # same as t_seconds for 5-orbit
        gmat_pos_km = gmat_data[:, 1:4]
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
    epoch_dt = datetime(year, 1, 1) + timedelta(days=day_frac - 1.0)

    if long_arc:
        # Train Neural ODE on full 7-day data
        print(f"  Training Neural ODE on first {TRAIN_FRAC*100:.0f}% of data...")
        ode_pred, n_train, model, norm = train_neural_ode_on_gmat(
            gmat_data, a_km, cd_a_over_m
        )
        pinn_pred_km = ode_pred[:, :3]
        # For inference timing, we need a tensor of all times
        t_all_tensor = torch.tensor(
            t_seconds[:, None], dtype=torch.float64, device=DEVICE
        )
    else:
        # Train FourierPINN (5-orbit mode, unchanged)
        print(f"  Training PINN on first {TRAIN_FRAC*100:.0f}% of window data...")
        pinn_pred_km, n_train, model, t_all_tensor, norm = train_pinn_on_gmat(
            gmat_data, a_km,
        )

    # 2. Propagate SGP4 from TLE epoch at the ORIGINAL times
    print(f"  Propagating SGP4...")
    sgp4_pos_km = propagate_sgp4_j2000(line1, line2, t_seconds_orig, epoch_dt)

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

    # --- Max Position Error (test region) ---
    pinn_max_err = float(np.max(pinn_err[n_train:]))
    sgp4_max_err = float(np.max(sgp4_err[n_train:]))

    # --- Inference Time ---
    model.eval()
    n_timing_runs = 10

    if long_arc:
        # Neural ODE: time a test-region integration
        state0_timing = torch.tensor(
            gmat_data[n_train - 1, 1:7], dtype=torch.float64, device=DEVICE
        )
        t0_timing = float(t_seconds[n_train - 1])
        # Use a subset of test points for timing (full integration is slow)
        t_timing_sub = torch.tensor(
            t_seconds[n_train:min(n_train + 500, len(t_seconds))],
            dtype=torch.float64, device=DEVICE
        )
        with torch.no_grad():
            _ = model.integrate_to_times(state0_timing, t0_timing,
                                         t_timing_sub, dt=NODE_DT)
        pinn_times = []
        for _ in range(min(n_timing_runs, 3)):  # fewer runs (integration is slow)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.integrate_to_times(state0_timing, t0_timing,
                                             t_timing_sub, dt=NODE_DT)
            pinn_times.append((time.perf_counter() - t0) * 1e3)
        # Scale to full test region
        scale = len(t_seconds[n_train:]) / len(t_timing_sub)
        pinn_inference_ms = float(np.median(pinn_times) * scale)
    else:
        # FourierPINN: time a forward pass on the full dataset
        with torch.no_grad():
            _ = model(t_all_tensor)  # warm-up
        pinn_times = []
        for _ in range(n_timing_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(t_all_tensor)
            pinn_times.append((time.perf_counter() - t0) * 1e3)
        pinn_inference_ms = float(np.median(pinn_times))

    # SGP4: time propagation of all points
    sat_sgp4 = Satrec.twoline2rv(line1, line2)
    N_pts = len(t_seconds)
    jd_array = np.full(N_pts, sat_sgp4.jdsatepoch)
    fr_array = sat_sgp4.jdsatepochF + t_seconds / 86400.0
    _ = sat_sgp4.sgp4_array(jd_array, fr_array)  # warm-up
    sgp4_times = []
    for _ in range(n_timing_runs):
        sat_sgp4 = Satrec.twoline2rv(line1, line2)
        jd_array = np.full(N_pts, sat_sgp4.jdsatepoch)
        fr_array = sat_sgp4.jdsatepochF + t_seconds / 86400.0
        t0 = time.perf_counter()
        _ = sat_sgp4.sgp4_array(jd_array, fr_array)
        sgp4_times.append((time.perf_counter() - t0) * 1e3)
    sgp4_inference_ms = float(np.median(sgp4_times))

    # GMAT: report execution time from metadata (not real-time runnable)
    gmat_inference_ms = meta.get("gmat_runtime_ms", None)

    # --- Max Energy Drift ---
    # Specific orbital energy: E = v^2/2 - mu/r
    # Approximate velocity via finite differences on the GMAT time grid
    dt = float(t_seconds[1] - t_seconds[0])

    def _energy_drift(pos_km_arr):
        vel = np.diff(pos_km_arr, axis=0) / dt  # (N-1, 3)
        v2 = np.sum(vel ** 2, axis=1)
        r = np.linalg.norm(pos_km_arr[:-1], axis=1)
        energy = 0.5 * v2 - MU / r
        return float(np.max(np.abs(energy - energy[0])))

    pinn_energy_drift = _energy_drift(pinn_pred_km)
    sgp4_energy_drift = _energy_drift(sgp4_pos_km)
    gmat_energy_drift = _energy_drift(gmat_pos_km)

    method = "NODE" if long_arc else "PINN"
    print(f"  {method} test RMSE:  {pinn_test_rmse:.2f} km")
    print(f"  SGP4 test RMSE:  {sgp4_test_rmse:.2f} km")
    print(f"  {method} max error:  {pinn_max_err:.2f} km")
    print(f"  SGP4 max error:  {sgp4_max_err:.2f} km")
    print(f"  {method} inference:  {pinn_inference_ms:.2f} ms")
    print(f"  SGP4 inference:  {sgp4_inference_ms:.2f} ms")
    print(f"  Energy drift  {method}={pinn_energy_drift:.2e}  "
          f"SGP4={sgp4_energy_drift:.2e}  GMAT={gmat_energy_drift:.2e}")

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
        "pinn_max_err_km": round(pinn_max_err, 4),
        "sgp4_max_err_km": round(sgp4_max_err, 4),
        "pinn_inference_ms": round(pinn_inference_ms, 2),
        "sgp4_inference_ms": round(sgp4_inference_ms, 2),
        "gmat_inference_ms": gmat_inference_ms,
        "pinn_energy_drift": pinn_energy_drift,
        "sgp4_energy_drift": sgp4_energy_drift,
        "gmat_energy_drift": gmat_energy_drift,
        "pinn_improvement_over_sgp4_pct": round(improvement, 2),
    }

    # Save per-satellite result
    os.makedirs("data/gmat_results", exist_ok=True)
    result_suffix = "_comparison_7day" if long_arc else "_comparison"
    result_path = f"data/gmat_results/{norad_id}{result_suffix}.json"
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
    parser.add_argument(
        "--long-arc", action="store_true",
        help="Use 7-day GMAT data (*_7day.npy) instead of 5-orbit data"
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

    arc_label = "7-day (Neural ODE)" if args.long_arc else "5-orbit"
    print(f"  Satellites: {len(satellites)}")
    print(f"  Data: {arc_label} GMAT trajectories")
    print(f"  Train fraction: {TRAIN_FRAC*100:.0f}%")

    all_results = []

    for idx, sat in enumerate(satellites):
        print(f"\n{'=' * 70}")
        print(f"  [{idx+1}/{len(satellites)}] {sat.name} (NORAD {sat.norad_id})")
        print(f"{'=' * 70}")

        result = compare_satellite(
            sat.norad_id, sat.name, sat.orbit_type,
            cd_a_over_m=sat.cd_a_over_m,
            long_arc=args.long_arc,
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

    # --- ISEF Results Table ---
    pinn_max_errs = np.array([r["pinn_max_err_km"] for r in all_results])
    sgp4_max_errs = np.array([r["sgp4_max_err_km"] for r in all_results])
    pinn_inf_ms = np.array([r["pinn_inference_ms"] for r in all_results])
    sgp4_inf_ms = np.array([r["sgp4_inference_ms"] for r in all_results])
    pinn_edrift = np.array([r["pinn_energy_drift"] for r in all_results])
    sgp4_edrift = np.array([r["sgp4_energy_drift"] for r in all_results])
    gmat_edrift = np.array([r["gmat_energy_drift"] for r in all_results])

    print(f"\n\n{'=' * 70}")
    print("  ISEF RESULTS TABLE  (N={} satellite average)".format(len(all_results)))
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<30s}  {'SGP4':>12s}  {'PINN':>12s}  {'GMAT':>12s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"  {'Position RMSE (km)':<30s}  "
          f"{np.mean(sgp4_rmses):>12.2f}  "
          f"{np.mean(pinn_rmses):>12.2f}  "
          f"{'0.00':>12s}")
    print(f"  {'Max Position Error (km)':<30s}  "
          f"{np.mean(sgp4_max_errs):>12.2f}  "
          f"{np.mean(pinn_max_errs):>12.2f}  "
          f"{'0.00':>12s}")
    print(f"  {'Inference Time (ms)':<30s}  "
          f"{np.mean(sgp4_inf_ms):>12.2f}  "
          f"{np.mean(pinn_inf_ms):>12.2f}  "
          f"{'N/A':>12s}")
    print(f"  {'Max Energy Drift (km²/s²)':<30s}  "
          f"{np.mean(sgp4_edrift):>12.2e}  "
          f"{np.mean(pinn_edrift):>12.2e}  "
          f"{np.mean(gmat_edrift):>12.2e}")

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
        "pinn_mean_max_err_km": round(float(np.mean(pinn_max_errs)), 4),
        "sgp4_mean_max_err_km": round(float(np.mean(sgp4_max_errs)), 4),
        "pinn_mean_inference_ms": round(float(np.mean(pinn_inf_ms)), 2),
        "sgp4_mean_inference_ms": round(float(np.mean(sgp4_inf_ms)), 2),
        "pinn_mean_energy_drift": float(np.mean(pinn_edrift)),
        "sgp4_mean_energy_drift": float(np.mean(sgp4_edrift)),
        "gmat_mean_energy_drift": float(np.mean(gmat_edrift)),
        "t_statistic": round(float(t_stat), 4),
        "p_value_one_sided": float(p_one),
        "cohens_d": round(float(cohens_d), 4),
        "significant_at_005": bool(p_one < 0.05),
        "results": all_results,
    }

    os.makedirs("data/gmat_results", exist_ok=True)
    suffix = "_7day" if args.long_arc else ""
    summary_path = f"data/gmat_results/hypothesis_test_summary{suffix}.json"
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
