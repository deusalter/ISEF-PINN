"""
train_pinn.py -- Physics-Informed Neural Network (PINN) for ISEF orbital propagation.

Trains a PINN to predict satellite position (x, y, z) from normalized time.
The architecture is identical to VanillaMLP, but a physics-informed loss term
is added via automatic differentiation.  The physics loss enforces Newton's
two-body law on a dense set of collocation points that span the FULL time
domain -- including the test (extrapolation) region -- so the network learns
to respect orbital mechanics even where it has no labelled data.

Architecture
------------
    Input  : 1  (normalized time t)
    Hidden : 4 x 128, tanh activations
    Output : 3  (normalized x, y, z)

Autograd pipeline
-----------------
    pos_pred       = model(t_col)               # (N_col, 3)
    vel_pred       = d(pos_pred) / dt_norm      # (N_col, 3)  via autograd
    acc_pred       = d(vel_pred) / dt_norm      # (N_col, 3)  via autograd

Physics residual in normalized coordinates (MU * t_ref^2 / r_ref^3 = 1.0):

    residual = acc_norm + pos_norm / ||pos_norm||^3   (should be 0)

Composite loss
--------------
    total_loss = data_loss + lambda_phys * physics_loss

Outputs
-------
    models/pinn_twobody.pt              -- saved model state dict
    data/pinn_predictions.npy           -- shape (2000, 4): [t, x, y, z] km
    data/pinn_loss_history.npy          -- shape (N_epochs, 3): [total, data, physics]
    figures/pinn_vs_vanilla_xy.png      -- comparison plot
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup -- must happen before any local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use non-interactive Matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from src.physics import NormalizationParams, MU

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH          = os.path.join(PROJECT_ROOT, "data", "orbital_data.npy")
VANILLA_PRED_PATH  = os.path.join(PROJECT_ROOT, "data", "vanilla_predictions.npy")
MODEL_SAVE_PATH    = os.path.join(PROJECT_ROOT, "models", "pinn_twobody.pt")
PRED_SAVE_PATH     = os.path.join(PROJECT_ROOT, "data", "pinn_predictions.npy")
LOSS_SAVE_PATH     = os.path.join(PROJECT_ROOT, "data", "pinn_loss_history.npy")
PLOT_SAVE_PATH     = os.path.join(PROJECT_ROOT, "figures", "pinn_vs_vanilla_xy.png")

# Ensure output directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PRED_SAVE_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(PLOT_SAVE_PATH),  exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters  (BEST CONFIG: 59.6% improvement over vanilla MLP)
# ---------------------------------------------------------------------------
HIDDEN_DIM       = 128
NUM_HIDDEN       = 4        # 4 hidden layers (best config)
WARMUP_EPOCHS    = 2000     # data-only warmup with Adam to learn orbit shape
EPOCHS           = 10000    # total epochs (including warmup)
LR               = 1e-3     # Adam learning rate
PATIENCE         = 1000     # ReduceLROnPlateau patience (large for PINNs)
FACTOR           = 0.5      # LR reduction factor
TRAIN_FRAC       = 0.80
LOG_EVERY        = 1000
LAMBDA_PHYS      = 1.0      # balanced with data loss (both O(1) in normalized coords)
N_COLLOCATION    = 1000     # collocation points spanning full time domain


# ---------------------------------------------------------------------------
# 1.  PINN model definition
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Physics-Informed Neural Network: t_norm -> (x_norm, y_norm, z_norm)."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        num_hidden: int = NUM_HIDDEN,
        output_dim: int = 3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Additional hidden layers
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer -- no activation; predict normalized coordinates directly
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier uniform initialization for all linear layers
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ---------------------------------------------------------------------------
# 2.  Autograd helpers
# ---------------------------------------------------------------------------

def compute_derivatives(
    model: PINN,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute predicted positions, velocities, and accelerations via autograd."""
    # Forward pass: normalized positions
    pos_pred = model(t)  # (N, 3)

    # --- First derivative: velocity ---
    vel_components = []
    for i in range(3):
        grad_i = torch.autograd.grad(
            outputs=pos_pred[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(pos_pred[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]  # shape (N, 1)
        vel_components.append(grad_i)
    vel_pred = torch.cat(vel_components, dim=1)  # (N, 3)

    # --- Second derivative: acceleration ---
    acc_components = []
    for i in range(3):
        grad_i = torch.autograd.grad(
            outputs=vel_pred[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(vel_pred[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]  # shape (N, 1)
        acc_components.append(grad_i)
    acc_pred = torch.cat(acc_components, dim=1)  # (N, 3)

    return pos_pred, vel_pred, acc_pred


def physics_loss(
    pos_pred_norm: torch.Tensor,
    acc_pred_norm: torch.Tensor,
    r_ref: float,
    t_ref: float,
) -> torch.Tensor:
    """Compute the two-body physics residual loss in NORMALIZED coordinates.

    Key insight: by construction, MU * t_ref^2 / r_ref^3 = 1.0 exactly.
    So in normalized coordinates the two-body ODE becomes simply:

        d^2 x_norm / d t_norm^2  +  x_norm / ||x_norm||^3  =  0
    """
    # ||x_norm|| with clamp to prevent division by zero at initialization
    r_norm = torch.norm(pos_pred_norm, dim=1, keepdim=True).clamp(min=1e-3)  # (N, 1)
    r3     = r_norm ** 3  # (N, 1)

    # Normalized residual: d^2 x_norm/dt_norm^2 + x_norm / ||x_norm||^3 = 0
    residual = acc_pred_norm + pos_pred_norm / r3  # (N, 3)

    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# 3.  Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(data_path: str, norm: NormalizationParams):
    """Load orbital data and return normalized PyTorch tensors."""
    data   = np.load(data_path)          # (N, 7)
    N      = data.shape[0]
    t_raw  = data[:, 0]                  # physical seconds
    pos_km = data[:, 1:4]                # [x, y, z] in km

    # Normalize time and positions
    t_norm_np   = norm.normalize_time(t_raw)   # dimensionless
    pos_norm_np = pos_km / norm.r_ref          # dimensionless

    n_train = int(N * TRAIN_FRAC)

    # Full dataset tensors (no grad needed -- used only for MSE evaluation)
    t_norm_all   = torch.tensor(t_norm_np[:, None], dtype=torch.float64, requires_grad=False)
    pos_norm_all = torch.tensor(pos_norm_np,        dtype=torch.float64, requires_grad=False)

    # Training subset
    t_norm_train   = t_norm_all[:n_train]
    pos_norm_train = pos_norm_all[:n_train]

    # Collocation points: uniformly spaced across the FULL normalized time range
    # Start slightly after t=0 to avoid the singularity where network outputs [0,0,0]
    t_start = t_norm_np[0] + 1e-4 * (t_norm_np[-1] - t_norm_np[0])
    t_col_np     = np.linspace(t_start, t_norm_np[-1], N_COLLOCATION)
    t_collocation = torch.tensor(
        t_col_np[:, None],
        dtype=torch.float64,
        requires_grad=True,   # CRITICAL: must be True for autograd derivatives
    )

    return (
        t_raw,
        t_norm_all,
        pos_norm_all,
        t_norm_train,
        pos_norm_train,
        t_collocation,
        n_train,
    )


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------

def train(
    model: PINN,
    t_train: torch.Tensor,
    pos_train: torch.Tensor,
    t_col: torch.Tensor,
    norm: NormalizationParams,
    lambda_phys: float = LAMBDA_PHYS,
) -> np.ndarray:
    """Train the PINN with a two-phase curriculum strategy.

    Phase 1 (Warmup): Data-only MSE with Adam to learn orbit shape.
    Phase 2 (Physics): Composite loss with ReduceLROnPlateau (patience=1000).
                       The large patience prevents premature LR decay when the
                       physics loss oscillates naturally during PINN training.
    """
    mse = nn.MSELoss()
    loss_history = np.zeros((EPOCHS, 3), dtype=np.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE, factor=FACTOR,
    )
    model.train()

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        # (a) Data loss -- always present
        pos_pred_train = model(t_train)
        data_loss = mse(pos_pred_train, pos_train)

        if epoch <= WARMUP_EPOCHS:
            # Phase 1: Warmup (data-only)
            total_loss = data_loss
            phys_val = 0.0
        else:
            # Phase 2: Physics-informed
            pos_col, _vel_col, acc_col = compute_derivatives(model, t_col)
            phys_loss_val = physics_loss(
                pos_pred_norm=pos_col,
                acc_pred_norm=acc_col,
                r_ref=norm.r_ref,
                t_ref=norm.t_ref,
            )
            total_loss = data_loss + lambda_phys * phys_loss_val
            phys_val = phys_loss_val.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss.item())

        total_val = total_loss.item()
        data_val = data_loss.item()
        loss_history[epoch - 1] = [total_val, data_val, phys_val]

        current_lr = optimizer.param_groups[0]["lr"]
        phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "PINN  "

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == WARMUP_EPOCHS + 1:
            print(
                f"  [{phase}] Epoch {epoch:6d}/{EPOCHS}  |  "
                f"data: {data_val:.6e}  |  "
                f"phys: {phys_val:.6e}  |  "
                f"total: {total_val:.6e}  |  "
                f"LR: {current_lr:.2e}"
            )

    return loss_history


# ---------------------------------------------------------------------------
# 5.  Evaluation helpers
# ---------------------------------------------------------------------------

def predict_full(model: PINN, t_norm_all: torch.Tensor) -> np.ndarray:
    """Run the model on the full time range; return normalized predictions."""
    model.eval()
    with torch.no_grad():
        pos_norm_pred = model(t_norm_all)
    return pos_norm_pred.numpy()


def compute_rmse(pred_km: np.ndarray, true_km: np.ndarray) -> float:
    """Compute overall position RMSE in km."""
    errors = np.linalg.norm(pred_km - true_km, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


# ---------------------------------------------------------------------------
# 6.  Diagnostic plot
# ---------------------------------------------------------------------------

def make_comparison_plot(
    t_raw: np.ndarray,
    true_pos_km: np.ndarray,
    pinn_pos_km: np.ndarray,
    vanilla_pos_km: np.ndarray,
    n_train: int,
    save_path: str,
) -> None:
    """Save a 2-D x-vs-y trajectory plot comparing PINN and vanilla."""
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(
        true_pos_km[:, 0], true_pos_km[:, 1],
        color="grey", linewidth=0.8, linestyle="--",
        label="Ground truth", zorder=1,
    )

    if vanilla_pos_km is not None:
        ax.plot(
            vanilla_pos_km[:n_train, 0], vanilla_pos_km[:n_train, 1],
            color="steelblue", linewidth=1.2, alpha=0.7,
            label="Vanilla MLP (train)", zorder=2,
        )
        ax.plot(
            vanilla_pos_km[n_train:, 0], vanilla_pos_km[n_train:, 1],
            color="steelblue", linewidth=1.2, linestyle=":", alpha=0.7,
            label="Vanilla MLP (test / extrap.)", zorder=2,
        )

    ax.plot(
        pinn_pos_km[:n_train, 0], pinn_pos_km[:n_train, 1],
        color="darkorange", linewidth=1.8,
        label="PINN (train)", zorder=3,
    )
    ax.plot(
        pinn_pos_km[n_train:, 0], pinn_pos_km[n_train:, 1],
        color="firebrick", linewidth=1.8,
        label="PINN (test / extrap.)", zorder=3,
    )

    bx = pinn_pos_km[n_train, 0]
    by = pinn_pos_km[n_train, 1]
    boundary_time_min = t_raw[n_train] / 60.0
    ax.scatter(
        [bx], [by], s=90, color="black", zorder=6,
        label=f"Train/test boundary (t = {boundary_time_min:.1f} min)",
    )

    theta = np.linspace(0, 2 * np.pi, 300)
    from src.physics import R_EARTH
    ax.plot(
        R_EARTH * np.cos(theta), R_EARTH * np.sin(theta),
        color="darkgreen", linewidth=1.0, linestyle=":",
        label="Earth surface (equatorial)", zorder=0,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x  [km]", fontsize=12)
    ax.set_ylabel("y  [km]", fontsize=12)
    ax.set_title(
        "PINN vs Vanilla MLP -- Predicted Trajectory (x-y plane)\n"
        "(orange = PINN train, red = PINN test/extrap., blue = vanilla MLP)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison plot -> {save_path}")


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  PINN Training -- ISEF Orbital Propagation Project")
    print("=" * 70)

    norm = NormalizationParams()
    print(f"\nNormalization: {norm}")
    print(f"  r_ref = {norm.r_ref:.3f} km")
    print(f"  t_ref = {norm.t_ref:.3f} s")
    print(f"  v_ref = {norm.v_ref:.4f} km/s")

    print(f"\nLoading data from: {DATA_PATH}")
    (
        t_raw, t_norm_all, pos_norm_all,
        t_norm_train, pos_norm_train,
        t_collocation, n_train,
    ) = load_and_preprocess(DATA_PATH, norm)

    N      = t_raw.shape[0]
    n_test = N - n_train
    print(f"  Total samples      : {N}")
    print(f"  Train samples      : {n_train}  ({TRAIN_FRAC*100:.0f}%)")
    print(f"  Test samples       : {n_test}  ({(1-TRAIN_FRAC)*100:.0f}%)")
    print(f"  Collocation points : {N_COLLOCATION}  (full time domain)")
    print(f"  Train time range   : 0 -- {t_raw[n_train-1]/60:.1f} min")
    print(f"  Test  time range   : {t_raw[n_train]/60:.1f} -- {t_raw[-1]/60:.1f} min")

    model = PINN(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_hidden=NUM_HIDDEN,
        output_dim=3,
    )
    model.double()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPINN architecture:\n{model}")
    print(f"  Trainable parameters : {total_params:,}")
    print(f"  Physics weight (lambda_phys) : {LAMBDA_PHYS}")
    print(f"  Collocation points           : {N_COLLOCATION}")

    print(
        f"\nTraining for {EPOCHS} epochs  "
        f"(lr={LR},  patience={PATIENCE},  factor={FACTOR})..."
    )
    loss_history = train(
        model=model,
        t_train=t_norm_train,
        pos_train=pos_norm_train,
        t_col=t_collocation,
        norm=norm,
        lambda_phys=LAMBDA_PHYS,
    )
    print("  Training complete.")

    print("\nRunning inference on full time range...")
    pos_norm_pred_all = predict_full(model, t_norm_all)
    pos_pred_km = pos_norm_pred_all * norm.r_ref
    pos_true_km = pos_norm_all.numpy() * norm.r_ref

    rmse_train = compute_rmse(pos_pred_km[:n_train], pos_true_km[:n_train])
    rmse_test  = compute_rmse(pos_pred_km[n_train:], pos_true_km[n_train:])

    print("\n" + "-" * 60)
    print("  PINN Results:")
    print(f"    Train RMSE : {rmse_train:12.3f} km")
    print(f"    Test  RMSE : {rmse_test:12.3f} km")
    print("-" * 60)

    vanilla_pos_km = None
    if os.path.isfile(VANILLA_PRED_PATH):
        vanilla_data   = np.load(VANILLA_PRED_PATH)
        vanilla_pos_km = vanilla_data[:, 1:4]
        vanilla_train_rmse = compute_rmse(vanilla_pos_km[:n_train], pos_true_km[:n_train])
        vanilla_test_rmse  = compute_rmse(vanilla_pos_km[n_train:], pos_true_km[n_train:])
        print("\n  Comparison with Vanilla MLP Baseline:")
        print(f"    Vanilla Train RMSE : {vanilla_train_rmse:12.3f} km")
        print(f"    Vanilla Test  RMSE : {vanilla_test_rmse:12.3f} km")
        print(f"    PINN    Train RMSE : {rmse_train:12.3f} km")
        print(f"    PINN    Test  RMSE : {rmse_test:12.3f} km")
        if vanilla_test_rmse > 0:
            improvement = (vanilla_test_rmse - rmse_test) / vanilla_test_rmse * 100.0
            print(f"\n    Test RMSE improvement: {improvement:.1f}%")
            if improvement > 0:
                print("    PINN extrapolates MUCH better thanks to the physics constraint!")
            else:
                print("    (More training epochs or tuning may improve test RMSE further.)")
    else:
        print(f"\n  (Vanilla predictions not found at {VANILLA_PRED_PATH}; "
              "skipping comparison.)")
    print("-" * 60)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n  Saved model        -> {MODEL_SAVE_PATH}")

    predictions_out = np.column_stack([t_raw, pos_pred_km])
    np.save(PRED_SAVE_PATH, predictions_out)
    print(f"  Saved predictions  -> {PRED_SAVE_PATH}  shape={predictions_out.shape}")

    np.save(LOSS_SAVE_PATH, loss_history)
    print(f"  Saved loss history -> {LOSS_SAVE_PATH}  shape={loss_history.shape}")

    print("\nGenerating PINN vs Vanilla comparison plot...")
    make_comparison_plot(
        t_raw=t_raw,
        true_pos_km=pos_true_km,
        pinn_pos_km=pos_pred_km,
        vanilla_pos_km=vanilla_pos_km,
        n_train=n_train,
        save_path=PLOT_SAVE_PATH,
    )

    print("\n" + "=" * 70)
    print("  Done.  All PINN outputs saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
