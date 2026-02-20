"""
Batch training on real satellite orbits (SGP4 data) -- ISEF PINN Project
========================================================================

For each satellite in the catalog, trains four models:
  1. Vanilla MLP    -- plain tanh NN, data-only
  2. Fourier NN     -- Fourier-featured NN, data-only
  3. Fourier PINN   -- Fourier-featured NN with J2-perturbed physics
  4. Fourier PINN   -- Fourier-featured NN with J2 + atmospheric drag physics

Data: loaded from data/real_orbits/{norad_id}.npy  (shape 5000x7)
Meta: loaded from data/real_orbits/{norad_id}_meta.json

Results saved to data/real_results/{norad_id}_results.json

Usage:
  python train_real_orbits.py              # train all satellites in catalog
  python train_real_orbits.py --sat 25544  # train a single satellite (ISS)
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

from src.physics import J2, R_EARTH, NormalizationParams
from src.atmosphere import drag_acceleration_torch
from satellite_catalog import get_catalog, get_by_norad_id

# -- Hyperparameters ----------------------------------------------------------

TOTAL_EPOCHS   = 5000
WARMUP_EPOCHS  = 1500    # data-only warmup for PINN
N_FREQ         = 6       # Fourier frequencies
HIDDEN         = 64      # hidden layer width
LAYERS         = 3       # hidden layers
LR_MAX         = 1e-3
LR_MIN         = 1e-5
N_COL          = 400     # collocation points
TRAIN_FRAC     = 0.20    # first 20% = ~1 orbit
GRAD_CLIP      = 1.0
LAMBDA_PHYS    = 0.005   # physics loss weight -- very gentle for SGP4 data


# -- Architecture (identical to train_pinn_j2.py) -----------------------------

class FourierPINN(nn.Module):
    """Fourier-featured neural network for orbital prediction."""

    def __init__(self, n_freq=N_FREQ, hidden=HIDDEN, n_layers=LAYERS):
        super().__init__()
        self.n_freq = n_freq
        # Purely periodic encoding (NO raw t) -- critical for extreme
        # extrapolation. Raw t would let tanh neurons produce non-periodic
        # outputs far beyond the training window (t goes 5x past training).
        input_dim = 2 * n_freq
        self.register_buffer(
            "freqs", torch.arange(1, n_freq + 1, dtype=torch.float64)
        )
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, t):
        wt = t * self.freqs
        return torch.cat([torch.sin(wt), torch.cos(wt)], dim=1)

    def forward(self, t):
        return self.net(self.encode(t))


class VanillaMLP(nn.Module):
    """Plain tanh MLP baseline (no Fourier features)."""

    def __init__(self, hidden=64, n_layers=LAYERS):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


# -- Physics loss (J2 in normalized coords, per-satellite R_NORM) -------------

def compute_j2_physics_loss(model, t_col, R_NORM):
    """J2-perturbed physics residual in normalized coordinates."""
    pos = model(t_col)
    ones = torch.ones(t_col.shape[0], dtype=torch.float64)

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

    # Two-body gravity (normalized)
    x_n = pos[:, 0:1]
    y_n = pos[:, 1:2]
    z_n = pos[:, 2:3]

    r = torch.norm(pos, dim=1, keepdim=True).clamp(min=1e-3)
    r3 = r ** 3
    r5 = r ** 5

    gravity = pos / r3

    # J2 perturbation (normalized)
    z_over_r_sq = (z_n / r) ** 2
    j2_coeff = -1.5 * J2 * (R_NORM ** 2) / r5

    a_j2_x = j2_coeff * x_n * (1.0 - 5.0 * z_over_r_sq)
    a_j2_y = j2_coeff * y_n * (1.0 - 5.0 * z_over_r_sq)
    a_j2_z = j2_coeff * z_n * (3.0 - 5.0 * z_over_r_sq)
    a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

    # Residual: acc + gravity - a_J2 = 0
    residual = acc + gravity - a_j2

    return torch.mean(residual ** 2)


def compute_j2_drag_physics_loss(model, t_col, R_NORM, norm, cd_a_over_m):
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
    cd_a_over_m   : float or scalar tensor, ballistic coeff Cd*A/m  [m^2/kg]
    """
    pos = model(t_col)
    ones = torch.ones(t_col.shape[0], dtype=torch.float64)

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

    gravity = pos / r3

    # J2 perturbation (normalized)
    z_over_r_sq = (z_n / r) ** 2
    j2_coeff = -1.5 * J2 * (R_NORM ** 2) / r5

    a_j2_x = j2_coeff * x_n * (1.0 - 5.0 * z_over_r_sq)
    a_j2_y = j2_coeff * y_n * (1.0 - 5.0 * z_over_r_sq)
    a_j2_z = j2_coeff * z_n * (3.0 - 5.0 * z_over_r_sq)
    a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

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

    # Full residual: acc + gravity - a_j2 - a_drag_normalized = 0
    residual = acc + gravity - a_j2 - a_drag_normalized

    return torch.mean(residual ** 2)


# -- Training function (adapted from train_pinn_j2.py) ------------------------

def train_model(model, t_train, pos_train, t_col, t_all, pos_all_km,
                n_train, total_epochs, warmup_epochs, lam_phys,
                R_NORM, norm, use_physics=False, tag="Model",
                physics_loss_fn=None):
    """Train a model and return (best_test_rmse, best_train_rmse).

    Parameters
    ----------
    physics_loss_fn : callable or None
        If provided, used instead of the default compute_j2_physics_loss.
        Signature: physics_loss_fn(model, t_col) -> scalar tensor.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    cosine_fn = lambda ep: (
        LR_MIN / LR_MAX
        + (1.0 - LR_MIN / LR_MAX)
        * 0.5 * (1.0 + math.cos(math.pi * ep / total_epochs))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_fn)
    mse = nn.MSELoss()

    best_test_rmse = float("inf")
    best_train_rmse = float("inf")
    best_state = None
    refreshed = False

    # Default physics loss: J2 only
    if physics_loss_fn is None:
        physics_loss_fn = lambda m, tc: compute_j2_physics_loss(m, tc, R_NORM)

    model.train()
    for ep in range(1, total_epochs + 1):
        optimizer.zero_grad()

        # Data loss
        pred = model(t_train)
        dl = mse(pred, pos_train)

        # Physics loss
        do_physics = use_physics and ep > warmup_epochs
        if do_physics:
            pl = physics_loss_fn(model, t_col)
            total = dl + lam_phys * pl
            pv = pl.item()
        else:
            total = dl
            pv = 0.0

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # Fresh optimizer at warmup->physics transition
        if use_physics and ep == warmup_epochs and not refreshed:
            refreshed = True
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX * 0.5)
            remaining = total_epochs - ep
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=LR_MIN
            )

        # Best model tracking
        if ep % 500 == 0 or ep == total_epochs:
            model.eval()
            with torch.no_grad():
                pred_km = model(t_all).numpy() * norm.r_ref
            err = np.linalg.norm(pred_km - pos_all_km, axis=1)
            test_rmse = float(np.sqrt(np.mean(err[n_train:] ** 2)))
            train_rmse = float(np.sqrt(np.mean(err[:n_train] ** 2)))
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_train_rmse = train_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

        # Print progress
        if ep % 2500 == 0 or ep == 1:
            lr = optimizer.param_groups[0]["lr"]
            phase = "DATA" if not do_physics else "PINN"
            print(
                f"  [{tag:>6s}|{phase:>4s}] {ep:5d}/{total_epochs}  "
                f"d={dl.item():.4e}  p={pv:.4e}  "
                f"lr={lr:.2e}  best_test={best_test_rmse:.1f}km",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_test_rmse, best_train_rmse


# -- Main ---------------------------------------------------------------------

def train_satellite(norad_id, name, orbit_type, cd_a_over_m=0.022, verbose=True):
    """Train all four models for one satellite. Returns results dict or None."""

    data_path = f"data/real_orbits/{norad_id}.npy"
    meta_path = f"data/real_orbits/{norad_id}_meta.json"

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

    # Per-satellite normalization
    norm = NormalizationParams(r_ref=a_km)
    R_NORM = R_EARTH / a_km

    t_norm_np = t_raw / norm.t_ref
    pos_norm_np = pos_km / norm.r_ref

    n_train = int(N * TRAIN_FRAC)
    n_test = N - n_train

    if verbose:
        print(f"\n  Normalization: {norm}")
        print(f"  R_NORM = R_EARTH / a_km = {R_NORM:.6f}")
        print(f"  Total: {N} pts | Train: {n_train} ({TRAIN_FRAC*100:.0f}%) | "
              f"Test: {n_test} ({(1-TRAIN_FRAC)*100:.0f}%)")
        print(f"  t_norm: [{t_norm_np[0]:.4f}, {t_norm_np[-1]:.4f}]")

    # Tensors
    t_train = torch.tensor(t_norm_np[:n_train, None], dtype=torch.float64)
    pos_train = torch.tensor(pos_norm_np[:n_train], dtype=torch.float64)
    t_all = torch.tensor(t_norm_np[:, None], dtype=torch.float64)

    # Collocation points in TRAINING region only -- avoids physics model
    # mismatch from fighting the true trajectory in the test region.
    # The Fourier + secular encoding handles test-region extrapolation.
    t_col = torch.linspace(
        float(t_norm_np[0]) + 0.001,
        float(t_norm_np[n_train - 1]),
        N_COL,
        dtype=torch.float64,
    ).unsqueeze(1).requires_grad_(True)

    # -- Train Vanilla MLP ----------------------------------------------------
    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  Training Vanilla MLP (data-only, no Fourier, no physics)...")
        print(f"  {'─' * 60}")

    torch.manual_seed(42)
    vanilla = VanillaMLP().double()
    van_test_rmse, van_train_rmse = train_model(
        vanilla, t_train, pos_train, t_col, t_all, pos_km,
        n_train, TOTAL_EPOCHS, TOTAL_EPOCHS, 0.0,
        R_NORM, norm,
        use_physics=False, tag="VAN",
    )

    # -- Train Fourier NN (data-only) -----------------------------------------
    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  Training Fourier NN (data-only, no physics)...")
        print(f"  {'─' * 60}")

    torch.manual_seed(42)
    fourier_nn = FourierPINN().double()
    nn_test_rmse, nn_train_rmse = train_model(
        fourier_nn, t_train, pos_train, t_col, t_all, pos_km,
        n_train, TOTAL_EPOCHS, TOTAL_EPOCHS, 0.0,
        R_NORM, norm,
        use_physics=False, tag="F-NN",
    )

    # -- Train Fourier PINN (data + J2 physics) -------------------------------
    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  Training Fourier PINN (data + J2 physics, lambda={LAMBDA_PHYS})...")
        print(f"  {'─' * 60}")

    torch.manual_seed(42)
    fourier_pinn = FourierPINN().double()
    pinn_test_rmse, pinn_train_rmse = train_model(
        fourier_pinn, t_train, pos_train, t_col, t_all, pos_km,
        n_train, TOTAL_EPOCHS, WARMUP_EPOCHS, LAMBDA_PHYS,
        R_NORM, norm,
        use_physics=True, tag="F-PINN",
    )

    # -- Train Fourier PINN (data + J2 + Drag physics) -------------------------
    if verbose:
        print(f"\n  {'─' * 60}")
        print(f"  Training Fourier PINN (J2+Drag, Cd*A/m={cd_a_over_m}, "
              f"lambda={LAMBDA_PHYS})...")
        print(f"  {'─' * 60}")

    torch.manual_seed(42)
    fourier_pinn_drag = FourierPINN().double()
    j2drag_loss_fn = lambda m, tc: compute_j2_drag_physics_loss(
        m, tc, R_NORM, norm, cd_a_over_m
    )
    drag_test_rmse, drag_train_rmse = train_model(
        fourier_pinn_drag, t_train, pos_train, t_col, t_all, pos_km,
        n_train, TOTAL_EPOCHS, WARMUP_EPOCHS, LAMBDA_PHYS,
        R_NORM, norm,
        use_physics=True, tag="J2+D",
        physics_loss_fn=j2drag_loss_fn,
    )

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
        "cd_a_over_m": cd_a_over_m,
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
    os.makedirs("data/real_results", exist_ok=True)
    result_path = f"data/real_results/{norad_id}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n  Results saved -> {result_path}")
        print(f"  {'Model':<24s}  {'Train RMSE':>12s}  {'Test RMSE':>12s}")
        print(f"  {'─' * 24}  {'─' * 12}  {'─' * 12}")
        print(f"  {'Vanilla MLP':<24s}  {van_train_rmse:>10.2f} km  {van_test_rmse:>10.2f} km")
        print(f"  {'Fourier NN':<24s}  {nn_train_rmse:>10.2f} km  {nn_test_rmse:>10.2f} km")
        print(f"  {'Fourier PINN (J2)':<24s}  {pinn_train_rmse:>10.2f} km  {pinn_test_rmse:>10.2f} km")
        print(f"  {'Fourier PINN (J2+Drag)':<24s}  {drag_train_rmse:>10.2f} km  {drag_test_rmse:>10.2f} km")
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
    args = parser.parse_args()

    print("=" * 70)
    print("  Real Satellite Orbits -- Fourier PINN Batch Training")
    print("=" * 70)

    n_params = sum(p.numel() for p in FourierPINN().double().parameters())
    print(f"\n  FourierPINN: N_FREQ={N_FREQ}, {HIDDEN}x{LAYERS} tanh, "
          f"{n_params:,} params")
    print(f"  Collocation: {N_COL} points | lambda_phys={LAMBDA_PHYS}")
    print(f"  Epochs: {TOTAL_EPOCHS} (warmup: {WARMUP_EPOCHS})")
    print(f"  Train fraction: {TRAIN_FRAC*100:.0f}%")

    os.makedirs("data/real_results", exist_ok=True)

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
            f"{'PINN(J2+D)':>12s}  {'J2 Imp':>8s}  {'J2+D Imp':>8s}"
        )
        print(
            f"  {'─' * 7}  {'─' * 22}  {'─' * 16}  "
            f"{'─' * 10}  {'─' * 10}  {'─' * 10}  "
            f"{'─' * 12}  {'─' * 8}  {'─' * 8}"
        )
        for r in all_results:
            print(
                f"  {r['norad_id']:>7d}  {r['name']:<22s}  {r['orbit_type']:<16s}  "
                f"{r['vanilla_test_rmse_km']:>8.1f}km  "
                f"{r['fourier_nn_test_rmse_km']:>8.1f}km  "
                f"{r['fourier_pinn_test_rmse_km']:>8.1f}km  "
                f"{r['j2drag_pinn_test_rmse_km']:>10.1f}km  "
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
        aggregate_path = "data/real_results/summary.json"
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
        print("  Make sure data files exist in data/real_orbits/")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
