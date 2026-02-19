"""
train_baseline.py -- Vanilla MLP baseline for ISEF PINN orbital propagation.

Trains a pure data-driven MLP (no physics loss) to predict satellite
position (x, y, z) from normalized time.  The deliberate goal is to
demonstrate that a vanilla network without physics constraints fails to
extrapolate beyond the training window -- the "straw man" for the PINN
comparison.

Architecture
------------
    Input  : 1  (normalized time t)
    Hidden : 3 x 64, tanh activations
    Output : 3  (normalized x, y, z)

Outputs
-------
    models/vanilla_mlp.pt               -- saved model state dict
    data/vanilla_predictions.npy        -- shape (2000, 4): [t, x, y, z] km
    data/vanilla_loss_history.npy       -- training MSE per epoch
    figures/vanilla_baseline_xy.png     -- diagnostic 2-D trajectory plot
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup -- must happen before any local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use non-interactive Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.physics import NormalizationParams

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH       = os.path.join(PROJECT_ROOT, "data", "orbital_data.npy")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "vanilla_mlp.pt")
PRED_SAVE_PATH  = os.path.join(PROJECT_ROOT, "data", "vanilla_predictions.npy")
LOSS_SAVE_PATH  = os.path.join(PROJECT_ROOT, "data", "vanilla_loss_history.npy")
PLOT_SAVE_PATH  = os.path.join(PROJECT_ROOT, "figures", "vanilla_baseline_xy.png")

# Ensure output directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PRED_SAVE_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(PLOT_SAVE_PATH),  exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_DIM  = 64
NUM_HIDDEN  = 3
BATCH_SIZE  = 256
EPOCHS      = 5000
LR          = 1e-3
SCHED_PATIENCE = 500
SCHED_FACTOR   = 0.5
TRAIN_FRAC  = 0.80
LOG_EVERY   = 500


# ---------------------------------------------------------------------------
# 1.  Model definition
# ---------------------------------------------------------------------------

class VanillaMLP(nn.Module):
    """Pure data-driven MLP: t_norm -> (x_norm, y_norm, z_norm).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the time input (1).
    hidden_dim : int
        Width of each hidden layer.
    num_hidden : int
        Number of hidden layers.
    output_dim : int
        Dimensionality of the position output (3).
    """

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

        # Hidden layers
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer (no activation -- predict normalized coordinates directly)
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
        """Forward pass.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1), dtype float64
            Normalized time values.

        Returns
        -------
        torch.Tensor, shape (N, 3), dtype float64
            Predicted normalized positions [x_norm, y_norm, z_norm].
        """
        return self.net(t)


# ---------------------------------------------------------------------------
# 2.  Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(data_path: str, norm: NormalizationParams):
    """Load orbital data and return normalized PyTorch tensors.

    Parameters
    ----------
    data_path : str
        Path to the .npy file with shape (N, 7): [t, x, y, z, vx, vy, vz].
    norm : NormalizationParams
        Normalization object from src.physics.

    Returns
    -------
    t_all_raw : np.ndarray, shape (N,)
        Raw physical times in seconds.
    t_norm_all : torch.Tensor, shape (N, 1), float64
        Normalized time for the full dataset.
    pos_norm_all : torch.Tensor, shape (N, 3), float64
        Normalized positions for the full dataset.
    t_norm_train : torch.Tensor, shape (N_train, 1), float64
    pos_norm_train : torch.Tensor, shape (N_train, 3), float64
    t_norm_test : torch.Tensor, shape (N_test, 1), float64
    pos_norm_test : torch.Tensor, shape (N_test, 3), float64
    n_train : int
        Number of training samples.
    """
    data = np.load(data_path)                        # (N, 7)
    N = data.shape[0]

    t_raw  = data[:, 0]                              # physical seconds
    pos_km = data[:, 1:4]                            # [x, y, z] in km

    # Normalize time and positions using NormalizationParams
    t_norm_np   = norm.normalize_time(t_raw)         # dimensionless
    pos_norm_np = pos_km / norm.r_ref                # dimensionless (same as normalize_state for positions)

    # Train / test split (first 80% / last 20% -- no shuffle to preserve temporal order)
    n_train = int(N * TRAIN_FRAC)

    # Convert to float64 tensors; requires_grad=False for the baseline
    t_norm_tensor   = torch.tensor(t_norm_np[:, None], dtype=torch.float64, requires_grad=False)
    pos_norm_tensor = torch.tensor(pos_norm_np,         dtype=torch.float64, requires_grad=False)

    t_norm_train   = t_norm_tensor[:n_train]
    pos_norm_train = pos_norm_tensor[:n_train]

    t_norm_test   = t_norm_tensor[n_train:]
    pos_norm_test = pos_norm_tensor[n_train:]

    return (
        t_raw,
        t_norm_tensor,
        pos_norm_tensor,
        t_norm_train,
        pos_norm_train,
        t_norm_test,
        pos_norm_test,
        n_train,
    )


# ---------------------------------------------------------------------------
# 3.  Training loop
# ---------------------------------------------------------------------------

def train(
    model: VanillaMLP,
    t_train: torch.Tensor,
    pos_train: torch.Tensor,
) -> list[float]:
    """Train the vanilla MLP with Adam + ReduceLROnPlateau.

    Parameters
    ----------
    model : VanillaMLP
    t_train : torch.Tensor, shape (N_train, 1)
    pos_train : torch.Tensor, shape (N_train, 3)

    Returns
    -------
    loss_history : list[float]
        MSE loss value recorded after every epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHED_PATIENCE,
        factor=SCHED_FACTOR,
    )
    loss_fn = nn.MSELoss()

    # DataLoader for mini-batch training
    dataset    = TensorDataset(t_train, pos_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_history: list[float] = []

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for t_batch, pos_batch in dataloader:
            optimizer.zero_grad()
            pos_pred = model(t_batch)
            loss     = loss_fn(pos_pred, pos_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        # Step the scheduler on the epoch-averaged training loss
        scheduler.step(avg_loss)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d}/{EPOCHS}  |  "
                f"Train MSE: {avg_loss:.6e}  |  "
                f"LR: {current_lr:.2e}"
            )

    return loss_history


# ---------------------------------------------------------------------------
# 4.  Evaluation helpers
# ---------------------------------------------------------------------------

def predict_full(model: VanillaMLP, t_norm_all: torch.Tensor) -> np.ndarray:
    """Run the model on the full time range; return normalized predictions.

    Parameters
    ----------
    model : VanillaMLP
    t_norm_all : torch.Tensor, shape (N, 1)

    Returns
    -------
    np.ndarray, shape (N, 3)
        Normalized predicted positions.
    """
    model.eval()
    with torch.no_grad():
        pos_norm_pred = model(t_norm_all)
    return pos_norm_pred.numpy()


def compute_rmse(pred_km: np.ndarray, true_km: np.ndarray) -> float:
    """Compute overall position RMSE in km.

    Parameters
    ----------
    pred_km : np.ndarray, shape (N, 3)
    true_km : np.ndarray, shape (N, 3)

    Returns
    -------
    float
        RMS of the Euclidean position errors [km].
    """
    errors = np.linalg.norm(pred_km - true_km, axis=1)   # (N,) per-sample 3-D error
    return float(np.sqrt(np.mean(errors ** 2)))


# ---------------------------------------------------------------------------
# 5.  Diagnostic plot
# ---------------------------------------------------------------------------

def make_xy_plot(
    t_raw: np.ndarray,
    true_pos_km: np.ndarray,
    pred_pos_km: np.ndarray,
    n_train: int,
    save_path: str,
) -> None:
    """Save a 2-D x-vs-y trajectory plot with train/test boundary marked.

    Parameters
    ----------
    t_raw : np.ndarray, shape (N,)
        Physical times in seconds (used only for the boundary label).
    true_pos_km : np.ndarray, shape (N, 3)
        Ground-truth positions [km].
    pred_pos_km : np.ndarray, shape (N, 3)
        Predicted positions [km].
    n_train : int
        Index separating training from test samples.
    save_path : str
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Ground truth (full orbit -- thin grey)
    ax.plot(
        true_pos_km[:, 0],
        true_pos_km[:, 1],
        color="grey",
        linewidth=0.8,
        linestyle="--",
        label="Ground truth",
        zorder=1,
    )

    # Predicted -- training region
    ax.plot(
        pred_pos_km[:n_train, 0],
        pred_pos_km[:n_train, 1],
        color="steelblue",
        linewidth=1.5,
        label="Predicted (train)",
        zorder=2,
    )

    # Predicted -- test region
    ax.plot(
        pred_pos_km[n_train:, 0],
        pred_pos_km[n_train:, 1],
        color="tomato",
        linewidth=1.5,
        label="Predicted (test / extrapolation)",
        zorder=2,
    )

    # Mark the train/test boundary point on the trajectory
    bx = pred_pos_km[n_train, 0]
    by = pred_pos_km[n_train, 1]
    boundary_time_min = t_raw[n_train] / 60.0
    ax.scatter(
        [bx], [by],
        s=80,
        color="black",
        zorder=5,
        label=f"Train/test boundary (t = {boundary_time_min:.1f} min)",
    )

    # Earth circle for context
    theta = np.linspace(0, 2 * np.pi, 300)
    from src.physics import R_EARTH
    ax.plot(
        R_EARTH * np.cos(theta),
        R_EARTH * np.sin(theta),
        color="darkgreen",
        linewidth=1.0,
        linestyle=":",
        label="Earth surface (equatorial)",
        zorder=0,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x  [km]", fontsize=12)
    ax.set_ylabel("y  [km]", fontsize=12)
    ax.set_title(
        "Vanilla MLP Baseline -- Predicted vs True Trajectory (x-y plane)\n"
        "(blue = train region, red = test / extrapolation region)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved trajectory plot -> {save_path}")


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Vanilla MLP Baseline Training -- ISEF PINN Project")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Normalization parameters (default: 400 km LEO reference orbit)
    # ------------------------------------------------------------------
    norm = NormalizationParams()
    print(f"\nNormalization: {norm}")

    # ------------------------------------------------------------------
    # Load and preprocess data
    # ------------------------------------------------------------------
    print(f"\nLoading data from: {DATA_PATH}")
    (
        t_raw,
        t_norm_all,
        pos_norm_all,
        t_norm_train,
        pos_norm_train,
        t_norm_test,
        pos_norm_test,
        n_train,
    ) = load_and_preprocess(DATA_PATH, norm)

    N = t_raw.shape[0]
    n_test = N - n_train
    print(f"  Total samples : {N}")
    print(f"  Train samples : {n_train}  ({TRAIN_FRAC*100:.0f}%)")
    print(f"  Test  samples : {n_test}  ({(1-TRAIN_FRAC)*100:.0f}%)")
    print(f"  Train time range : 0 -- {t_raw[n_train-1]/60:.1f} min")
    print(f"  Test  time range : {t_raw[n_train]/60:.1f} -- {t_raw[-1]/60:.1f} min")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = VanillaMLP(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_hidden=NUM_HIDDEN,
        output_dim=3,
    )
    model.double()   # enforce float64 throughout

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel architecture:\n{model}")
    print(f"  Trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {EPOCHS} epochs  (batch={BATCH_SIZE}, lr={LR})...")
    loss_history = train(model, t_norm_train, pos_norm_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # Predict on full time range (train + test)
    # ------------------------------------------------------------------
    print("\nRunning inference on full time range...")
    pos_norm_pred_all = predict_full(model, t_norm_all)   # (N, 3), normalized

    # Denormalize to physical km
    pos_pred_km = pos_norm_pred_all * norm.r_ref          # (N, 3)
    pos_true_km = pos_norm_all.numpy() * norm.r_ref       # (N, 3)

    # ------------------------------------------------------------------
    # Compute RMSE on train and test subsets
    # ------------------------------------------------------------------
    rmse_train = compute_rmse(pos_pred_km[:n_train], pos_true_km[:n_train])
    rmse_test  = compute_rmse(pos_pred_km[n_train:], pos_true_km[n_train:])

    print("\n" + "-" * 55)
    print(f"  Train RMSE : {rmse_train:10.3f} km")
    print(f"  Test  RMSE : {rmse_test:10.3f} km")
    print(
        "  (Test RMSE >> Train RMSE is expected: vanilla NNs\n"
        "   cannot extrapolate orbital mechanics without physics.)"
    )
    print("-" * 55)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n  Saved model       -> {MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Save predictions: shape (N, 4) = [t_seconds, x_km, y_km, z_km]
    # ------------------------------------------------------------------
    predictions_out = np.column_stack([t_raw, pos_pred_km])   # (N, 4)
    np.save(PRED_SAVE_PATH, predictions_out)
    print(f"  Saved predictions -> {PRED_SAVE_PATH}  shape={predictions_out.shape}")

    # ------------------------------------------------------------------
    # Save loss history
    # ------------------------------------------------------------------
    loss_arr = np.array(loss_history, dtype=np.float64)
    np.save(LOSS_SAVE_PATH, loss_arr)
    print(f"  Saved loss history-> {LOSS_SAVE_PATH}  ({len(loss_arr)} values)")

    # ------------------------------------------------------------------
    # Diagnostic plot
    # ------------------------------------------------------------------
    print("\nGenerating diagnostic plot...")
    make_xy_plot(
        t_raw=t_raw,
        true_pos_km=pos_true_km,
        pred_pos_km=pos_pred_km,
        n_train=n_train,
        save_path=PLOT_SAVE_PATH,
    )

    print("\n" + "=" * 65)
    print("  Done.  All outputs saved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
