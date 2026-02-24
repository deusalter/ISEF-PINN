"""
evaluate.py -- Comprehensive evaluation & visualization for the ISEF PINN orbital propagation project.

Loads all saved data and models, computes all metrics, and generates every
plot needed for the ISEF poster board.  Designed to run after both
train_baseline.py and train_pinn.py have been executed.  Gracefully
degrades when individual files are missing -- a warning is printed and
the corresponding analysis section is skipped.

Data layouts (loaded from data/):
    orbital_data.npy          (N, 7) : [t, x, y, z, vx, vy, vz]  N=2000
    orbital_data_j2.npy       (N, 7) : [t, x, y, z, vx, vy, vz]  N=5000
    vanilla_predictions.npy   (N, 4) : [t, x, y, z]               km
    pinn_predictions.npy      (N, 4) : [t, x, y, z]               km
    vanilla_j2_predictions.npy (N,4) : [t, x, y, z]               km  (optional)
    pinn_j2_predictions.npy   (N, 4) : [t, x, y, z]               km  (optional)
    vanilla_loss_history.npy  (E,)   : MSE per epoch
    pinn_loss_history.npy     (E, 3) : [total, data, physics] per epoch

Usage:
    python3 evaluate.py
"""

# ---------------------------------------------------------------------------
# Backend must be set before any pyplot import
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import os
import sys
import timeit
import warnings

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: project root -> sys.path so local imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.physics import (
    hamiltonian_energy,
    MU,
    orbital_period,
    R_ORBIT,
    NormalizationParams,
)

from plotting import (
    plot_3d_comparison,
    plot_energy_conservation,
    plot_loss_convergence,
    plot_rmse_over_time,
    plot_error_distribution,
    create_rmse_table,
    plot_2d_trajectory_comparison,
)

from train_baseline import VanillaMLP
from train_pinn import PINN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Training fraction (two-body uses 0.80, J2 uses 0.20)
TRAIN_FRAC_TWOBODY: float = 0.80
TRAIN_FRAC_J2: float = 0.20

# ---------------------------------------------------------------------------
# Checkpoint times (seconds) used for the RMSE table
# ---------------------------------------------------------------------------
RMSE_CHECKPOINTS: tuple = (100, 1000, 5000, 10000)


# ===========================================================================
# Section 1 -- Utility helpers
# ===========================================================================

def warn_missing(path: str, context: str = "") -> None:
    """Print a clearly formatted warning when a required file is absent."""
    msg = f"[WARN] File not found: {path}"
    if context:
        msg += f"  ({context})"
    print(msg)


def try_load_npy(path: str, context: str = "") -> np.ndarray | None:
    """Load a .npy file; return None with a warning if the file is missing."""
    if not os.path.isfile(path):
        warn_missing(path, context)
        return None
    arr = np.load(path)
    print(f"[load] {os.path.basename(path):40s}  shape={arr.shape}")
    return arr


def compute_position_errors(pred_pos: np.ndarray, true_pos: np.ndarray) -> np.ndarray:
    """Return per-timestep Euclidean position error in km.

    Parameters
    ----------
    pred_pos : np.ndarray, shape (N, 3)
        Predicted positions [x, y, z] in km.
    true_pos : np.ndarray, shape (N, 3)
        Ground-truth positions [x, y, z] in km.

    Returns
    -------
    np.ndarray, shape (N,)
        ||pred - true|| at each timestep [km].
    """
    return np.linalg.norm(pred_pos - true_pos, axis=1)


def compute_rmse(errors: np.ndarray) -> float:
    """Root-mean-square of an error array."""
    return float(np.sqrt(np.mean(errors ** 2)))


# ===========================================================================
# Section 2 -- Hamiltonian / energy computation
# ===========================================================================

def _central_diff_velocity(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Estimate velocity via central finite differences.

    Uses central differences for interior points and one-sided (forward /
    backward) differences at the endpoints.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time stamps in seconds.
    positions : np.ndarray, shape (N, 3)
        Positions [x, y, z] in km.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Approximate velocities [vx, vy, vz] in km/s.
    """
    N = len(times)
    vel = np.empty_like(positions)

    # Central differences for interior points
    dt_forward  = times[2:] - times[1:-1]   # (N-2,)
    dt_backward = times[1:-1] - times[:-2]  # (N-2,)
    dt_total    = times[2:] - times[:-2]     # (N-2,) = t[i+1] - t[i-1]
    vel[1:-1] = (positions[2:] - positions[:-2]) / dt_total[:, None]

    # Forward difference at t=0
    vel[0] = (positions[1] - positions[0]) / (times[1] - times[0])

    # Backward difference at t=T
    vel[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])

    return vel


def compute_energy_from_states(times: np.ndarray, positions: np.ndarray,
                                velocities: np.ndarray | None = None) -> np.ndarray:
    """Compute specific orbital energy H = 0.5*v^2 - MU/r at every timestep.

    When velocities are provided (e.g., from the ground truth), they are used
    directly.  When they are None (e.g., for a prediction file that has only
    positions), velocity is estimated via central finite differences.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time stamps in seconds.
    positions : np.ndarray, shape (N, 3)
        Positions [x, y, z] in km.
    velocities : np.ndarray or None, shape (N, 3)
        True velocities [vx, vy, vz] in km/s.  Pass None to use finite
        differences from positions.

    Returns
    -------
    np.ndarray, shape (N,)
        Specific orbital energy [km^2/s^2] at each timestep.
    """
    if velocities is None:
        velocities = _central_diff_velocity(times, positions)

    r    = np.linalg.norm(positions, axis=1)                  # (N,)
    v_sq = np.sum(velocities ** 2, axis=1)                    # (N,)
    return 0.5 * v_sq - MU / r                                # (N,)


def energy_drift(energy: np.ndarray) -> float:
    """Return max |H(t) - H(0)| as the energy conservation metric."""
    return float(np.max(np.abs(energy - energy[0])))


# ===========================================================================
# Section 3 -- Model loading helpers
# ===========================================================================

def _infer_hidden_dim_num_hidden(state_dict: dict) -> tuple[int, int]:
    """Infer (hidden_dim, num_hidden) from a checkpoint's state dict keys.

    The sequential net has layers indexed 0, 2, 4, ... (linear layers with
    Tanh activations in between).  The last linear maps hidden -> output_dim.

    Returns
    -------
    (hidden_dim, num_hidden)
    """
    linear_keys = sorted(
        [k for k in state_dict if k.endswith(".weight") and "net." in k],
        key=lambda k: int(k.split(".")[1]),
    )
    # First linear: net.0.weight has shape (hidden_dim, input_dim)
    hidden_dim = state_dict[linear_keys[0]].shape[0]
    # Number of linear layers = len(linear_keys); last one is output layer.
    # num_hidden = total linears - 1 (output layer) = len(linear_keys) - 1
    num_hidden = len(linear_keys) - 1
    return hidden_dim, num_hidden


def load_vanilla_model(model_path: str) -> VanillaMLP | None:
    """Load a VanillaMLP from a checkpoint, inferring the architecture.

    Returns None if the file is absent.
    """
    if not os.path.isfile(model_path):
        warn_missing(model_path, "VanillaMLP model checkpoint")
        return None
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    hidden_dim, num_hidden = _infer_hidden_dim_num_hidden(sd)
    model = VanillaMLP(input_dim=1, hidden_dim=hidden_dim,
                       num_hidden=num_hidden, output_dim=3)
    model.double()
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] VanillaMLP loaded  hidden_dim={hidden_dim}  "
          f"num_hidden={num_hidden}  params={sum(p.numel() for p in model.parameters()):,}")
    return model


def load_pinn_model(model_path: str) -> PINN | None:
    """Load a PINN from a checkpoint, inferring the architecture.

    Returns None if the file is absent.
    """
    if not os.path.isfile(model_path):
        warn_missing(model_path, "PINN model checkpoint")
        return None
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    hidden_dim, num_hidden = _infer_hidden_dim_num_hidden(sd)
    model = PINN(input_dim=1, hidden_dim=hidden_dim,
                 num_hidden=num_hidden, output_dim=3)
    model.double()
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] PINN loaded        hidden_dim={hidden_dim}  "
          f"num_hidden={num_hidden}  params={sum(p.numel() for p in model.parameters()):,}")
    return model


# ===========================================================================
# Section 4 -- Latency benchmarking
# ===========================================================================

def benchmark_latency(
    model: torch.nn.Module | None,
    t_tensor: torch.Tensor,
    label: str,
    n_repeats: int = 50,
) -> float | None:
    """Time model inference on t_tensor and return latency in ms.

    Parameters
    ----------
    model : nn.Module or None
        The model to benchmark.  Returns None if model is None.
    t_tensor : torch.Tensor
        Input time tensor (already on CPU).
    label : str
        Human-readable name printed in the report.
    n_repeats : int
        Number of timing repetitions.

    Returns
    -------
    float or None
        Median inference latency in milliseconds, or None.
    """
    if model is None:
        return None

    model.eval()

    # Warm up JIT / kernel caches
    with torch.no_grad():
        for _ in range(5):
            model(t_tensor)

    times_ms = []
    for _ in range(n_repeats):
        t_start = timeit.default_timer()
        with torch.no_grad():
            model(t_tensor)
        t_end = timeit.default_timer()
        times_ms.append((t_end - t_start) * 1000.0)

    latency = float(np.median(times_ms))
    print(f"  {label:20s}  latency = {latency:.3f} ms  "
          f"(median of {n_repeats} runs, N={t_tensor.shape[0]} points)")
    return latency


# ===========================================================================
# Section 5 -- Core analysis: given ground-truth + predictions, do everything
# ===========================================================================

def run_analysis(
    label: str,
    gt_data: np.ndarray,
    vanilla_pred_data: np.ndarray | None,
    pinn_pred_data: np.ndarray | None,
    vanilla_loss_history: np.ndarray | None,
    pinn_loss_history: np.ndarray | None,
    vanilla_model: torch.nn.Module | None,
    pinn_model: torch.nn.Module | None,
    out_prefix: str,
    train_frac: float = 0.80,
) -> dict:
    """Run the full evaluation pipeline for one dataset (two-body or J2).

    Parameters
    ----------
    label : str
        Human-readable label, e.g. "Two-body" or "J2-perturbed".
    gt_data : np.ndarray, shape (N, 7)
        Ground-truth array [t, x, y, z, vx, vy, vz].
    vanilla_pred_data : np.ndarray or None, shape (N, 4)
        Vanilla predictions [t, x, y, z] in km.
    pinn_pred_data : np.ndarray or None, shape (N, 4)
        PINN predictions [t, x, y, z] in km.
    vanilla_loss_history : np.ndarray or None
        Shape (E,) with per-epoch MSE, or None.
    pinn_loss_history : np.ndarray or None
        Shape (E, 3) with [total, data, physics] per epoch, or None.
    vanilla_model : nn.Module or None
        Loaded VanillaMLP for latency benchmarking.
    pinn_model : nn.Module or None
        Loaded PINN for latency benchmarking.
    out_prefix : str
        Filename prefix for output figures, e.g. "" (empty) or "j2_".

    Returns
    -------
    dict
        Summary metrics for the printed report.
    """
    print()
    print("=" * 70)
    print(f"  Analysis: {label}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Unpack ground truth
    # ------------------------------------------------------------------
    times      = gt_data[:, 0].astype(np.float64)   # seconds
    pos_true   = gt_data[:, 1:4].astype(np.float64) # km  (N, 3)
    vel_true   = gt_data[:, 4:7].astype(np.float64) # km/s (N, 3)
    N          = len(times)
    n_train    = int(N * train_frac)

    # ------------------------------------------------------------------
    # Unpack predictions (positions only)
    # ------------------------------------------------------------------
    pos_vanilla: np.ndarray | None = None
    pos_pinn:    np.ndarray | None = None

    if vanilla_pred_data is not None:
        pos_vanilla = vanilla_pred_data[:, 1:4].astype(np.float64)  # km (N, 3)

    if pinn_pred_data is not None:
        pos_pinn = pinn_pred_data[:, 1:4].astype(np.float64)        # km (N, 3)

    # ------------------------------------------------------------------
    # Position errors (Euclidean distance, km)
    # ------------------------------------------------------------------
    errors_vanilla: np.ndarray | None = None
    errors_pinn:    np.ndarray | None = None

    if pos_vanilla is not None:
        errors_vanilla = compute_position_errors(pos_vanilla, pos_true)
        print(f"\n  Vanilla NN position errors:")
        print(f"    Mean : {np.mean(errors_vanilla):.4f} km")
        print(f"    Max  : {np.max(errors_vanilla):.4f} km")

    if pos_pinn is not None:
        errors_pinn = compute_position_errors(pos_pinn, pos_true)
        print(f"\n  PINN position errors:")
        print(f"    Mean : {np.mean(errors_pinn):.4f} km")
        print(f"    Max  : {np.max(errors_pinn):.4f} km")

    # ------------------------------------------------------------------
    # RMSE split: train vs test
    # ------------------------------------------------------------------
    metrics: dict = {}

    if errors_vanilla is not None:
        rmse_v_train = compute_rmse(errors_vanilla[:n_train])
        rmse_v_test  = compute_rmse(errors_vanilla[n_train:])
        metrics["vanilla_rmse_train"] = rmse_v_train
        metrics["vanilla_rmse_test"]  = rmse_v_test
        print(f"\n  Vanilla NN RMSE  -- train: {rmse_v_train:.4f} km  "
              f"| test: {rmse_v_test:.4f} km")
    else:
        metrics["vanilla_rmse_train"] = None
        metrics["vanilla_rmse_test"]  = None

    if errors_pinn is not None:
        rmse_p_train = compute_rmse(errors_pinn[:n_train])
        rmse_p_test  = compute_rmse(errors_pinn[n_train:])
        metrics["pinn_rmse_train"] = rmse_p_train
        metrics["pinn_rmse_test"]  = rmse_p_test
        print(f"  PINN RMSE        -- train: {rmse_p_train:.4f} km  "
              f"| test: {rmse_p_test:.4f} km")
    else:
        metrics["pinn_rmse_train"] = None
        metrics["pinn_rmse_test"]  = None

    # Improvement on test set
    if errors_vanilla is not None and errors_pinn is not None:
        rv_test = metrics["vanilla_rmse_test"]
        rp_test = metrics["pinn_rmse_test"]
        if rv_test > 0:
            improvement = (rv_test - rp_test) / rv_test * 100.0
        else:
            improvement = 0.0
        metrics["test_improvement_pct"] = improvement
        print(f"\n  Test RMSE improvement (PINN vs Vanilla): {improvement:.2f}%")
    else:
        metrics["test_improvement_pct"] = None

    # ------------------------------------------------------------------
    # Hamiltonian energy computation
    # ------------------------------------------------------------------
    print("\n  Computing Hamiltonian energy...")

    # Ground truth: use true velocities for exact energy
    energy_truth = compute_energy_from_states(times, pos_true, vel_true)
    drift_truth  = energy_drift(energy_truth)
    metrics["energy_drift_truth"] = drift_truth
    print(f"    Ground truth  H0={energy_truth[0]:.6f}  "
          f"drift={drift_truth:.6e} km^2/s^2")

    # Vanilla NN: finite-difference velocity from positions only
    energy_vanilla: np.ndarray | None = None
    if pos_vanilla is not None:
        energy_vanilla = compute_energy_from_states(times, pos_vanilla,
                                                     velocities=None)
        drift_vanilla  = energy_drift(energy_vanilla)
        metrics["energy_drift_vanilla"] = drift_vanilla
        print(f"    Vanilla NN    H0={energy_vanilla[0]:.6f}  "
              f"drift={drift_vanilla:.6e} km^2/s^2")
    else:
        metrics["energy_drift_vanilla"] = None

    # PINN: finite-difference velocity from positions only
    energy_pinn: np.ndarray | None = None
    if pos_pinn is not None:
        energy_pinn = compute_energy_from_states(times, pos_pinn,
                                                  velocities=None)
        drift_pinn  = energy_drift(energy_pinn)
        metrics["energy_drift_pinn"] = drift_pinn
        print(f"    PINN          H0={energy_pinn[0]:.6f}  "
              f"drift={drift_pinn:.6e} km^2/s^2")
    else:
        metrics["energy_drift_pinn"] = None

    # ------------------------------------------------------------------
    # Latency benchmarking
    # ------------------------------------------------------------------
    print("\n  Benchmarking inference latency...")
    norm     = NormalizationParams()
    t_norm   = norm.normalize_time(times)
    t_tensor = torch.tensor(t_norm[:, None], dtype=torch.float64)

    lat_vanilla = benchmark_latency(vanilla_model, t_tensor, "Vanilla NN")
    lat_pinn    = benchmark_latency(pinn_model,    t_tensor, "PINN")
    metrics["latency_vanilla_ms"] = lat_vanilla
    metrics["latency_pinn_ms"]    = lat_pinn

    # ------------------------------------------------------------------
    # Plot 1: 3D trajectory comparison
    # ------------------------------------------------------------------
    save_3d = os.path.join(FIGURES_DIR, f"{out_prefix}3d_comparison.png")
    if errors_vanilla is not None:
        # All plotting functions accept (N,3) or (N,6+) arrays
        plot_3d_comparison(
            ground_truth  = np.asarray(pos_true,   dtype=float),
            vanilla_pred  = np.asarray(pos_vanilla, dtype=float),
            pinn_pred     = np.asarray(pos_pinn,    dtype=float) if pos_pinn is not None else None,
            save_path     = save_3d,
            title         = f"Orbital Trajectory Comparison ({label})",
        )
    else:
        print(f"[skip] 3D comparison -- no vanilla predictions available")

    # ------------------------------------------------------------------
    # Plot 2: Energy conservation
    # ------------------------------------------------------------------
    save_energy = os.path.join(FIGURES_DIR, f"{out_prefix}energy_conservation.png")
    if energy_vanilla is not None:
        plot_energy_conservation(
            times         = np.asarray(times,          dtype=float),
            energy_truth  = np.asarray(energy_truth,   dtype=float),
            energy_vanilla= np.asarray(energy_vanilla, dtype=float),
            energy_pinn   = np.asarray(energy_pinn,    dtype=float) if energy_pinn is not None else None,
            save_path     = save_energy,
        )
    else:
        print(f"[skip] Energy conservation plot -- no vanilla predictions available")

    # ------------------------------------------------------------------
    # Plot 3: Loss convergence
    # ------------------------------------------------------------------
    save_loss = os.path.join(FIGURES_DIR, f"{out_prefix}loss_convergence.png")
    if vanilla_loss_history is not None and pinn_loss_history is not None:
        # pinn_loss_history shape: (E, 3) => columns [total, data, physics]
        pinn_lh = np.asarray(pinn_loss_history, dtype=float)
        if pinn_lh.ndim == 2 and pinn_lh.shape[1] >= 3:
            lh_pinn_data   = pinn_lh[:, 1]
            lh_pinn_physics = pinn_lh[:, 2]
        else:
            # Fallback: treat as single column (total), set physics=0
            warnings.warn("pinn_loss_history does not have 3 columns; "
                          "treating as total loss only.")
            lh_pinn_data    = pinn_lh.ravel()
            lh_pinn_physics = np.zeros_like(lh_pinn_data)

        plot_loss_convergence(
            loss_history_vanilla       = np.asarray(vanilla_loss_history, dtype=float).ravel(),
            loss_history_pinn_data     = lh_pinn_data,
            loss_history_pinn_physics  = lh_pinn_physics,
            save_path                  = save_loss,
        )
    else:
        missing = []
        if vanilla_loss_history is None:
            missing.append("vanilla_loss_history")
        if pinn_loss_history is None:
            missing.append("pinn_loss_history")
        print(f"[skip] Loss convergence plot -- missing: {', '.join(missing)}")

    # ------------------------------------------------------------------
    # Plot 4: RMSE over time
    # ------------------------------------------------------------------
    save_rmse = os.path.join(FIGURES_DIR, f"{out_prefix}rmse_over_time.png")
    if errors_vanilla is not None:
        plot_rmse_over_time(
            times          = np.asarray(times,          dtype=float),
            errors_vanilla = np.asarray(errors_vanilla, dtype=float),
            errors_pinn    = np.asarray(errors_pinn,    dtype=float) if errors_pinn is not None else None,
            save_path      = save_rmse,
            train_fraction = train_frac,
        )
    else:
        print(f"[skip] RMSE over time -- no vanilla predictions available")

    # ------------------------------------------------------------------
    # Plot 5: Error distribution histogram
    # ------------------------------------------------------------------
    save_hist = os.path.join(FIGURES_DIR, f"{out_prefix}error_distribution.png")
    if errors_vanilla is not None:
        plot_error_distribution(
            errors_vanilla = np.asarray(errors_vanilla, dtype=float),
            errors_pinn    = np.asarray(errors_pinn,    dtype=float) if errors_pinn is not None else None,
            save_path      = save_hist,
        )
    else:
        print(f"[skip] Error distribution -- no vanilla predictions available")

    # ------------------------------------------------------------------
    # Table 6: RMSE checkpoint table
    # ------------------------------------------------------------------
    save_table = os.path.join(FIGURES_DIR, f"{out_prefix}rmse_table.txt")
    if errors_vanilla is not None:
        create_rmse_table(
            times          = np.asarray(times,          dtype=float),
            errors_vanilla = np.asarray(errors_vanilla, dtype=float),
            errors_pinn    = np.asarray(errors_pinn,    dtype=float) if errors_pinn is not None else None,
            output_path    = save_table,
            checkpoints    = RMSE_CHECKPOINTS,
        )
    else:
        print(f"[skip] RMSE table -- no vanilla predictions available")

    # ------------------------------------------------------------------
    # Plot 7: 2D trajectory comparison
    # ------------------------------------------------------------------
    save_2d = os.path.join(FIGURES_DIR, f"{out_prefix}2d_comparison.png")
    if errors_vanilla is not None:
        plot_2d_trajectory_comparison(
            ground_truth   = np.asarray(pos_true,    dtype=float),
            vanilla_pred   = np.asarray(pos_vanilla, dtype=float),
            pinn_pred      = np.asarray(pos_pinn,    dtype=float) if pos_pinn is not None else None,
            save_path      = save_2d,
            train_fraction = train_frac,
        )
    else:
        print(f"[skip] 2D trajectory comparison -- no vanilla predictions available")

    return metrics


# ===========================================================================
# Section 6 -- Summary report printer
# ===========================================================================

def print_summary(label: str, metrics: dict) -> None:
    """Print a formatted ISEF-ready summary for a single dataset analysis."""
    sep = "=" * 70
    print()
    print(sep)
    print(f"  SUMMARY -- {label}")
    print(sep)

    # RMSE
    def _fmt(val):
        return f"{val:.4f} km" if val is not None else "N/A"

    print(f"\n  Position RMSE:")
    print(f"    Vanilla NN  -- train : {_fmt(metrics.get('vanilla_rmse_train'))}")
    print(f"    Vanilla NN  -- test  : {_fmt(metrics.get('vanilla_rmse_test'))}")
    print(f"    PINN        -- train : {_fmt(metrics.get('pinn_rmse_train'))}")
    print(f"    PINN        -- test  : {_fmt(metrics.get('pinn_rmse_test'))}")

    improvement = metrics.get("test_improvement_pct")
    if improvement is not None:
        direction = "improvement" if improvement >= 0 else "regression"
        print(f"\n  Test-set RMSE {direction}: {improvement:.2f}%")

    # Energy drift
    print(f"\n  Hamiltonian energy drift  (max |H - H0|):")
    drift_gt  = metrics.get("energy_drift_truth")
    drift_v   = metrics.get("energy_drift_vanilla")
    drift_p   = metrics.get("energy_drift_pinn")
    def _drift_fmt(val):
        return f"{val:.6e} km^2/s^2" if val is not None else "N/A"
    print(f"    Ground truth : {_drift_fmt(drift_gt)}")
    print(f"    Vanilla NN   : {_drift_fmt(drift_v)}")
    print(f"    PINN         : {_drift_fmt(drift_p)}")

    if drift_v is not None and drift_p is not None and drift_v > 0:
        energy_improvement = (drift_v - drift_p) / drift_v * 100.0
        print(f"    Energy conservation improvement (PINN vs Vanilla): "
              f"{energy_improvement:.2f}%")

    # Latency
    print(f"\n  Inference latency (2000 points, CPU, median of 50 runs):")
    lat_v = metrics.get("latency_vanilla_ms")
    lat_p = metrics.get("latency_pinn_ms")
    print(f"    Vanilla NN : {f'{lat_v:.3f} ms' if lat_v is not None else 'N/A'}")
    print(f"    PINN       : {f'{lat_p:.3f} ms' if lat_p is not None else 'N/A'}")

    print()
    print(sep)


# ===========================================================================
# Section 7 -- Main entry point
# ===========================================================================

def main() -> None:
    print()
    print("=" * 70)
    print("  ISEF PINN Orbital Propagation -- Comprehensive Evaluation")
    print("=" * 70)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Figures dir  : {FIGURES_DIR}")

    # -----------------------------------------------------------------------
    # 7.1  Load models (skip — we use saved predictions instead)
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Loading models (skipped — using saved predictions)...")
    vanilla_model = None
    pinn_model = None

    # -----------------------------------------------------------------------
    # 7.2  Load loss histories
    # -----------------------------------------------------------------------
    print("\n[Phase 2] Loading loss histories...")
    vanilla_loss_history = try_load_npy(
        os.path.join(DATA_DIR, "vanilla_loss_history.npy"),
        "vanilla per-epoch MSE loss")
    pinn_loss_history = try_load_npy(
        os.path.join(DATA_DIR, "pinn_loss_history.npy"),
        "PINN [total, data, physics] per-epoch losses")

    # -----------------------------------------------------------------------
    # 7.3  Two-body analysis
    # -----------------------------------------------------------------------
    print("\n[Phase 3] Loading two-body ground truth data...")
    gt_twobody = try_load_npy(
        os.path.join(DATA_DIR, "orbital_data.npy"),
        "two-body ground truth (2000 pts)")

    if gt_twobody is None:
        print("[ERROR] Cannot proceed without ground-truth data.  "
              "Run generate_data.py first.")
    else:
        print("\n[Phase 4] Loading two-body predictions...")
        vanilla_pred = try_load_npy(
            os.path.join(DATA_DIR, "vanilla_predictions.npy"),
            "Vanilla NN two-body predictions")
        pinn_pred = try_load_npy(
            os.path.join(DATA_DIR, "pinn_predictions.npy"),
            "PINN two-body predictions")

        print("\n[Phase 5] Running two-body analysis...")
        metrics_twobody = run_analysis(
            label               = "Two-body (Keplerian)",
            gt_data             = gt_twobody,
            vanilla_pred_data   = vanilla_pred,
            pinn_pred_data      = pinn_pred,
            vanilla_loss_history= vanilla_loss_history,
            pinn_loss_history   = pinn_loss_history,
            vanilla_model       = vanilla_model,
            pinn_model          = pinn_model,
            out_prefix          = "",   # saves to figures/3d_comparison.png etc.
            train_frac          = TRAIN_FRAC_TWOBODY,
        )
        print_summary("Two-body (Keplerian)", metrics_twobody)

    # -----------------------------------------------------------------------
    # 7.4  J2-perturbed analysis (optional -- graceful skip if absent)
    # -----------------------------------------------------------------------
    print("\n[Phase 6] Checking for J2-perturbed data...")
    gt_j2 = try_load_npy(
        os.path.join(DATA_DIR, "orbital_data_j2.npy"),
        "J2-perturbed ground truth (5000 pts)")

    if gt_j2 is not None:
        print("\n[Phase 7] Loading J2-perturbed predictions...")
        vanilla_j2_pred = try_load_npy(
            os.path.join(DATA_DIR, "vanilla_j2_predictions.npy"),
            "Vanilla NN J2 predictions")
        pinn_j2_pred = try_load_npy(
            os.path.join(DATA_DIR, "pinn_j2_predictions.npy"),
            "PINN J2 predictions")

        if vanilla_j2_pred is None and pinn_j2_pred is None:
            print("[INFO] No J2 prediction files found (vanilla_j2_predictions.npy "
                  "/ pinn_j2_predictions.npy).  Skipping J2 plot generation.")
        else:
            print("\n[Phase 8] Running J2-perturbed analysis...")
            metrics_j2 = run_analysis(
                label               = "J2-perturbed",
                gt_data             = gt_j2,
                vanilla_pred_data   = vanilla_j2_pred,
                pinn_pred_data      = pinn_j2_pred,
                vanilla_loss_history= None,  # separate J2 loss files not required
                pinn_loss_history   = None,
                vanilla_model       = None,  # model latency already reported above
                pinn_model          = None,
                out_prefix          = "j2_",  # saves to figures/j2_3d_comparison.png etc.
                train_frac          = TRAIN_FRAC_J2,
            )
            print_summary("J2-perturbed", metrics_j2)
    else:
        print("[INFO] J2 ground truth not found -- skipping J2 analysis.")

    # -----------------------------------------------------------------------
    # 7.5  Final file manifest
    # -----------------------------------------------------------------------
    print("\n[Phase 9] Output file manifest:")
    expected_files = [
        "3d_comparison.png",
        "energy_conservation.png",
        "loss_convergence.png",
        "rmse_over_time.png",
        "error_distribution.png",
        "rmse_table.txt",
        "2d_comparison.png",
    ]
    for fname in expected_files:
        fpath = os.path.join(FIGURES_DIR, fname)
        status = "OK  " if os.path.isfile(fpath) else "MISS"
        print(f"  [{status}]  {fpath}")

    print()
    print("=" * 70)
    print("  Evaluation complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
