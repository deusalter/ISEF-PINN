"""
J2-Perturbed Fourier PINN for Orbital Propagation -- ISEF Project
=================================================================

Extreme extrapolation experiment: train on 1 orbit (20%), predict 5 orbits.

Three models compared:
  1. Vanilla MLP    -- plain tanh NN, data-only
  2. Fourier NN     -- Fourier-featured NN, data-only
  3. Fourier PINN   -- Fourier-featured NN with J2-perturbed physics

J2 physics in normalized coordinates (MU * t_ref^2 / r_ref^3 = 1):
    d^2 r~ / dt~^2  =  -r~ / ||r~||^3  +  a_J2~

    R_norm   = R_EARTH / r_ref
    j2_coeff = -1.5 * J2 * R_norm^2 / r~^5
    a_J2_x~  = j2_coeff * x~ * (1 - 5*(z~/r~)^2)
    a_J2_y~  = j2_coeff * y~ * (1 - 5*(z~/r~)^2)
    a_J2_z~  = j2_coeff * z~ * (3 - 5*(z~/r~)^2)

Data: 5000 points, 5 orbits with J2 perturbation.
  Train: first 20% (~1 orbit), Test: last 80% (~4 orbits)
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from src.physics import J2, R_EARTH, R_ORBIT, NormalizationParams

# ── Reproducibility ──────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters ──────────────────────────────────────────────────
TOTAL_EPOCHS   = 5000
WARMUP_EPOCHS  = 1500    # data-only warmup for PINN
N_FREQ         = 4       # Fourier frequencies (low to avoid k^2 blowup)
HIDDEN         = 64      # hidden layer width (matches two-body)
LAYERS         = 3       # hidden layers
LR_MAX         = 1e-3
LR_MIN         = 1e-5
N_COL          = 400     # collocation points (periodic encoding makes dense coverage unnecessary)
TRAIN_FRAC     = 0.20    # first 20% = ~1 orbit
GRAD_CLIP      = 1.0
LAMBDA_PHYS    = 0.05    # slightly higher for J2 (physics matters more)

norm = NormalizationParams(r_ref=R_ORBIT)
R_NORM = R_EARTH / norm.r_ref  # normalized Earth radius (~0.941)


# ── Architecture ─────────────────────────────────────────────────────

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


# ── Physics loss ─────────────────────────────────────────────────────

def compute_j2_physics_loss(model, t_col):
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


# ── Training function ────────────────────────────────────────────────

def train_model(model, t_train, pos_train, t_col, t_all, pos_all_km,
                n_train, total_epochs, warmup_epochs, lam_phys,
                use_physics=False, tag="Model"):
    """Train a model and return (best_test_rmse, loss_history)."""

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    cosine_fn = lambda ep: (
        LR_MIN / LR_MAX
        + (1.0 - LR_MIN / LR_MAX)
        * 0.5 * (1.0 + math.cos(math.pi * ep / total_epochs))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_fn)
    mse = nn.MSELoss()

    best_test_rmse = float("inf")
    best_state = None
    loss_history = []
    refreshed = False

    model.train()
    for ep in range(1, total_epochs + 1):
        optimizer.zero_grad()

        # Data loss
        pred = model(t_train)
        dl = mse(pred, pos_train)

        # Physics loss
        do_physics = use_physics and ep > warmup_epochs
        if do_physics:
            pl = compute_j2_physics_loss(model, t_col)
            total = dl + lam_phys * pl
            pv = pl.item()
        else:
            total = dl
            pv = 0.0

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        loss_history.append([total.item(), dl.item(), pv])

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
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
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

    return best_test_rmse, np.array(loss_history)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  J2-Perturbed Fourier PINN -- ISEF Orbital Propagation")
    print("=" * 70)

    print(f"\nNormalization: {norm}")
    print(f"  R_norm = R_EARTH / r_ref = {R_NORM:.6f}")
    print(f"  J2     = {J2:.5e}")
    mu_check = 398600.4418 * norm.t_ref ** 2 / norm.r_ref ** 3
    print(f"  MU * t_ref^2 / r_ref^3 = {mu_check:.6f} (should be 1.0)")

    # ── Load J2 data ─────────────────────────────────────────────────
    data_path = "data/orbital_data_j2.npy"
    print(f"\nLoading J2 data from: {data_path}")
    data = np.load(data_path)        # (5000, 7)
    N = data.shape[0]
    t_raw = data[:, 0]
    pos_km = data[:, 1:4]

    t_norm_np = t_raw / norm.t_ref
    pos_norm_np = pos_km / norm.r_ref

    n_train = int(N * TRAIN_FRAC)
    n_test = N - n_train

    print(f"  Total : {N} points")
    print(f"  Train : {n_train} ({TRAIN_FRAC*100:.0f}% -- ~1 orbit)")
    print(f"  Test  : {n_test} ({(1-TRAIN_FRAC)*100:.0f}% -- ~4 orbits)")
    print(f"  t_norm: [{t_norm_np[0]:.4f}, {t_norm_np[-1]:.4f}]")

    # ── Tensors ──────────────────────────────────────────────────────
    t_train = torch.tensor(t_norm_np[:n_train, None], dtype=torch.float64)
    pos_train = torch.tensor(pos_norm_np[:n_train], dtype=torch.float64)
    t_all = torch.tensor(t_norm_np[:, None], dtype=torch.float64)

    t_col = torch.linspace(
        float(t_norm_np[0]) + 0.001,
        float(t_norm_np[-1]),
        N_COL,
        dtype=torch.float64,
    ).unsqueeze(1).requires_grad_(True)

    # ── Architecture info ────────────────────────────────────────────
    n_params = sum(p.numel() for p in FourierPINN().double().parameters())
    print(f"\n  FourierPINN: N_FREQ={N_FREQ}, {HIDDEN}x{LAYERS} tanh, "
          f"{n_params:,} params")
    print(f"  Collocation: {N_COL} points | lambda_phys={LAMBDA_PHYS}")
    print(f"  Epochs: {TOTAL_EPOCHS} (warmup: {WARMUP_EPOCHS})")

    # ── Load or train Vanilla MLP baseline ─────────────────────────────
    van_pred_path = "data/vanilla_j2_predictions.npy"
    if os.path.exists(van_pred_path):
        print(f"\n  Loading existing vanilla predictions: {van_pred_path}")
        van_pred_loaded = np.load(van_pred_path)
        van_pred_km = van_pred_loaded[:, 1:4]
        van_err = np.linalg.norm(van_pred_km - pos_km, axis=1)
        van_train_rmse = float(np.sqrt(np.mean(van_err[:n_train] ** 2)))
        van_test_rmse = float(np.sqrt(np.mean(van_err[n_train:] ** 2)))
        print(f"  Vanilla MLP: train={van_train_rmse:.1f} km, test={van_test_rmse:.1f} km")
        skip_vanilla = True
    else:
        print(f"\n{'─' * 70}")
        print("  Training Vanilla MLP (data-only, no Fourier, no physics)...")
        print(f"{'─' * 70}")
        torch.manual_seed(42)
        vanilla = VanillaMLP().double()
        van_rmse, van_loss = train_model(
            vanilla, t_train, pos_train, t_col, t_all, pos_km,
            n_train, TOTAL_EPOCHS, TOTAL_EPOCHS, 0.0,
            use_physics=False, tag="VAN",
        )
        skip_vanilla = False

    # ── Train Fourier NN (data-only) ─────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Training Fourier NN (data-only, no physics)...")
    print(f"{'─' * 70}")

    torch.manual_seed(42)
    fourier_nn = FourierPINN().double()
    nn_rmse, nn_loss = train_model(
        fourier_nn, t_train, pos_train, t_col, t_all, pos_km,
        n_train, TOTAL_EPOCHS, TOTAL_EPOCHS, 0.0,
        use_physics=False, tag="F-NN",
    )

    # ── Train Fourier PINN (data + J2 physics) ───────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Training Fourier PINN (data + J2 physics, lambda={LAMBDA_PHYS})...")
    print(f"{'─' * 70}")

    torch.manual_seed(42)
    fourier_pinn = FourierPINN().double()
    pinn_rmse, pinn_loss = train_model(
        fourier_pinn, t_train, pos_train, t_col, t_all, pos_km,
        n_train, TOTAL_EPOCHS, WARMUP_EPOCHS, LAMBDA_PHYS,
        use_physics=True, tag="F-PINN",
    )

    # ── Evaluate all three models ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  EVALUATION -- J2 Perturbed Orbit")
    print(f"{'=' * 70}")

    # Vanilla predictions (already computed or loaded above)
    if not skip_vanilla:
        vanilla.eval()
        with torch.no_grad():
            van_pred_km = vanilla(t_all).numpy() * norm.r_ref
        van_err = np.linalg.norm(van_pred_km - pos_km, axis=1)
    van_train = float(np.sqrt(np.mean(van_err[:n_train] ** 2)))
    van_test = float(np.sqrt(np.mean(van_err[n_train:] ** 2)))

    # Fourier NN predictions
    fourier_nn.eval()
    with torch.no_grad():
        nn_pred_km = fourier_nn(t_all).numpy() * norm.r_ref
    nn_err = np.linalg.norm(nn_pred_km - pos_km, axis=1)
    nn_train = float(np.sqrt(np.mean(nn_err[:n_train] ** 2)))
    nn_test = float(np.sqrt(np.mean(nn_err[n_train:] ** 2)))

    # Fourier PINN predictions
    fourier_pinn.eval()
    with torch.no_grad():
        pinn_pred_km = fourier_pinn(t_all).numpy() * norm.r_ref
    pinn_err = np.linalg.norm(pinn_pred_km - pos_km, axis=1)
    pinn_train = float(np.sqrt(np.mean(pinn_err[:n_train] ** 2)))
    pinn_test = float(np.sqrt(np.mean(pinn_err[n_train:] ** 2)))

    # Per-orbit breakdown
    pts_per_orbit = N // 5
    print(f"\n  Per-orbit RMSE (km):")
    print(f"  {'Orbit':>8s}  {'Type':>5s}  {'Vanilla':>10s}  {'Fourier NN':>12s}  {'Fourier PINN':>12s}")
    print(f"  {'─' * 8}  {'─' * 5}  {'─' * 10}  {'─' * 12}  {'─' * 12}")
    for i in range(5):
        s, e = i * pts_per_orbit, (i + 1) * pts_per_orbit
        v_r = float(np.sqrt(np.mean(van_err[s:e] ** 2)))
        n_r = float(np.sqrt(np.mean(nn_err[s:e] ** 2)))
        p_r = float(np.sqrt(np.mean(pinn_err[s:e] ** 2)))
        label = "TRAIN" if i == 0 else "TEST"
        print(f"  {i+1:>8d}  {label:>5s}  {v_r:>10.1f}  {n_r:>12.1f}  {p_r:>12.1f}")

    # Summary
    print(f"\n  {'Model':<20s}  {'Train RMSE':>12s}  {'Test RMSE':>12s}")
    print(f"  {'─' * 20}  {'─' * 12}  {'─' * 12}")
    print(f"  {'Vanilla MLP':<20s}  {van_train:>10.2f} km  {van_test:>10.2f} km")
    print(f"  {'Fourier NN':<20s}  {nn_train:>10.2f} km  {nn_test:>10.2f} km")
    print(f"  {'Fourier PINN (J2)':<20s}  {pinn_train:>10.2f} km  {pinn_test:>10.2f} km")

    nn_improv = (van_test - nn_test) / van_test * 100 if van_test > 0 else 0
    pinn_improv = (van_test - pinn_test) / van_test * 100 if van_test > 0 else 0
    pinn_vs_nn = (nn_test - pinn_test) / nn_test * 100 if nn_test > 0 else 0

    print(f"\n  Fourier NN  improvement over Vanilla: {nn_improv:+.1f}%")
    print(f"  Fourier PINN improvement over Vanilla: {pinn_improv:+.1f}%")
    print(f"  Fourier PINN improvement over Fourier NN: {pinn_vs_nn:+.1f}%")

    # ── Save outputs ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SAVING OUTPUTS")
    print(f"{'=' * 70}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # PINN J2 predictions
    pinn_preds = np.column_stack([t_raw, pinn_pred_km])
    np.save("data/pinn_j2_predictions.npy", pinn_preds)
    print(f"  Saved PINN J2 predictions -> data/pinn_j2_predictions.npy  {pinn_preds.shape}")

    # PINN J2 loss history
    np.save("data/pinn_j2_loss_history.npy", pinn_loss)
    print(f"  Saved PINN J2 loss history -> data/pinn_j2_loss_history.npy  {pinn_loss.shape}")

    # PINN J2 model
    torch.save(fourier_pinn.state_dict(), "models/pinn_j2.pt")
    print(f"  Saved PINN J2 model       -> models/pinn_j2.pt")

    # Vanilla J2 predictions
    if not skip_vanilla:
        van_preds = np.column_stack([t_raw, van_pred_km])
        np.save("data/vanilla_j2_predictions.npy", van_preds)
        print(f"  Saved Vanilla J2 preds    -> data/vanilla_j2_predictions.npy  {van_preds.shape}")
    else:
        print(f"  Vanilla J2 preds          -> already at data/vanilla_j2_predictions.npy")

    # ── Comparison plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # X-Y trajectory
    ax = axes[0]
    ax.plot(pos_km[:, 0], pos_km[:, 1], color="grey", lw=0.8, ls="--",
            label="J2 Ground Truth", zorder=1)
    ax.plot(van_pred_km[:, 0], van_pred_km[:, 1], "r:", lw=0.8, alpha=0.7,
            label=f"Vanilla MLP (test={van_test:.0f} km)", zorder=2)
    ax.plot(nn_pred_km[:, 0], nn_pred_km[:, 1], "m--", lw=1.0, alpha=0.7,
            label=f"Fourier NN (test={nn_test:.0f} km)", zorder=2)
    ax.plot(pinn_pred_km[:, 0], pinn_pred_km[:, 1], "g-", lw=1.5,
            label=f"Fourier PINN (test={pinn_test:.0f} km)", zorder=3)
    ax.scatter([pos_km[n_train, 0]], [pos_km[n_train, 1]],
               s=80, c="black", zorder=5, label="Train/Test boundary")

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_EARTH * np.cos(theta), R_EARTH * np.sin(theta),
            color="darkgreen", lw=0.8, ls=":", label="Earth", zorder=0)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("J2 Trajectory Comparison")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Error over time
    ax = axes[1]
    t_min = t_raw / 60.0
    ax.semilogy(t_min, van_err, "r-", alpha=0.5, lw=0.8, label="Vanilla MLP")
    ax.semilogy(t_min, nn_err, "m-", alpha=0.5, lw=0.8, label="Fourier NN")
    ax.semilogy(t_min, pinn_err, "g-", alpha=0.8, lw=1.2, label="Fourier PINN")
    ax.axvline(t_min[n_train], color="k", ls="--", alpha=0.5,
               label="Train/Test boundary")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title("J2 Position Error Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/pinn_j2_comparison_xy.png", dpi=150)
    plt.close()
    print(f"  Saved J2 comparison plot  -> outputs/pinn_j2_comparison_xy.png")

    print(f"\n{'=' * 70}")
    print("  Done. All J2 outputs saved.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
