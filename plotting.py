"""
plotting.py — Publication-quality visualization for the ISEF PINN orbital propagation project.

All plotting functions in this module are designed to produce high-resolution
figures suitable for an ISEF poster board.  Each function accepts NumPy arrays,
optionally handles the case where ``pinn_pred`` is ``None`` (Phase 2, before the
PINN is trained), saves the figure to ``save_path``, and calls
``tight_layout()`` before saving.

Units throughout:  km for position, km/s for velocity, s for time.

Colour convention
-----------------
GROUND_TRUTH : '#1f77b4'   (blue)   — high-accuracy RK8 numerical integrator
VANILLA_NN   : '#d62728'   (red)    — fully-connected NN without physics loss
PINN         : '#2ca02c'   (green)  — physics-informed neural network
EARTH        : '#4a7c59'   (muted green) — Earth sphere wireframe
"""

import os
import sys

# ---------------------------------------------------------------------------
# Matplotlib backend — must be set before importing pyplot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for 3-D projection

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.physics import R_EARTH, R_ORBIT, MU, orbital_period

# ---------------------------------------------------------------------------
# Global style — publication quality
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        # Font
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
        # Lines
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
GROUND_TRUTH: str = "#1f77b4"   # blue
VANILLA_NN: str   = "#d62728"   # red
PINN: str         = "#2ca02c"   # green
EARTH: str        = "#4a7c59"   # muted green

# Orbital period of the reference 400 km LEO (seconds)
_T_ORBIT: float = orbital_period(R_ORBIT)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _times_to_periods(times: np.ndarray) -> np.ndarray:
    """Convert a time array in seconds to units of orbital periods."""
    return np.asarray(times, dtype=float) / _T_ORBIT


def _equal_aspect_3d(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Force equal axis scaling on a 3-D Axes object."""
    all_pts = np.concatenate([x, y, z])
    max_range = (all_pts.max() - all_pts.min()) / 2.0
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _earth_wireframe_coords(n_lon: int = 40, n_lat: int = 20):
    """Return (x, y, z) meshgrids for an Earth-radius wireframe sphere."""
    u = np.linspace(0, 2 * np.pi, n_lon)
    v = np.linspace(0, np.pi, n_lat)
    ex = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ey = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ez = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    return ex, ey, ez


# ---------------------------------------------------------------------------
# 1.  plot_3d_comparison
# ---------------------------------------------------------------------------

def plot_3d_comparison(
    ground_truth: np.ndarray,
    vanilla_pred: np.ndarray,
    pinn_pred,
    save_path: str,
    title: str = "Orbital Trajectory Comparison",
) -> None:
    """Create a high-resolution 3-D trajectory comparison plot.

    Parameters
    ----------
    ground_truth : np.ndarray, shape (N, 3) or (N, 6+)
        Ground-truth positions [x, y, z] in km.  The first three columns are
        used; additional columns (velocities etc.) are ignored.
    vanilla_pred : np.ndarray, shape (N, 3) or (N, 6+)
        Vanilla NN predicted positions.
    pinn_pred : np.ndarray or None, shape (N, 3) or (N, 6+)
        PINN predicted positions.  Pass ``None`` to skip (Phase 2).
    save_path : str
        File path for the output PNG.
    title : str, optional
        Figure title.  Pass an empty string or ``None`` to suppress.
    """
    ground_truth = np.asarray(ground_truth)
    vanilla_pred = np.asarray(vanilla_pred) if vanilla_pred is not None else None

    # Extract positional columns
    gt_x, gt_y, gt_z = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # --- Trajectories ---
    ax.plot(gt_x, gt_y, gt_z,
            color=GROUND_TRUTH, linestyle="-", linewidth=1.8,
            label="Ground Truth (RK8)", zorder=4)

    if vanilla_pred is not None:
        vx, vy, vz = vanilla_pred[:, 0], vanilla_pred[:, 1], vanilla_pred[:, 2]
        ax.plot(vx, vy, vz,
                color=VANILLA_NN, linestyle=":", linewidth=1.8,
                label="Vanilla NN", zorder=3)

    if pinn_pred is not None:
        pinn_pred = np.asarray(pinn_pred)
        px, py, pz = pinn_pred[:, 0], pinn_pred[:, 1], pinn_pred[:, 2]
        ax.plot(px, py, pz,
                color=PINN, linestyle="--", linewidth=1.8,
                label="PINN", zorder=5)

    # --- Earth wireframe ---
    ex, ey, ez = _earth_wireframe_coords()
    ax.plot_wireframe(ex, ey, ez,
                      color=EARTH, alpha=0.20, linewidth=0.4,
                      label="Earth")

    # Mark start point on ground truth
    ax.scatter([gt_x[0]], [gt_y[0]], [gt_z[0]],
               color=GROUND_TRUTH, s=50, zorder=6)

    # --- Axes & labels ---
    ax.set_xlabel("X (km)", labelpad=8)
    ax.set_ylabel("Y (km)", labelpad=8)
    ax.set_zlabel("Z (km)", labelpad=8)
    if title:
        ax.set_title(title, pad=14)

    _equal_aspect_3d(ax, gt_x, gt_y, gt_z)
    ax.legend(loc="best", framealpha=0.85)

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plotting] 3D comparison saved -> {save_path}")


# ---------------------------------------------------------------------------
# 2.  plot_energy_conservation
# ---------------------------------------------------------------------------

def plot_energy_conservation(
    times: np.ndarray,
    energy_truth: np.ndarray,
    energy_vanilla: np.ndarray,
    energy_pinn,
    save_path: str,
) -> None:
    """Plot specific orbital energy (Hamiltonian) conservation over time.

    A well-behaved integrator / PINN should produce a nearly flat line;
    the Vanilla NN should show visible drift, demonstrating physics violation.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time stamps in seconds.
    energy_truth : np.ndarray, shape (N,)
        Ground-truth Hamiltonian  H = 0.5*v^2 - MU/r  [km^2/s^2].
    energy_vanilla : np.ndarray, shape (N,)
        Vanilla NN Hamiltonian values.
    energy_pinn : np.ndarray or None, shape (N,)
        PINN Hamiltonian values.  Pass ``None`` to skip.
    save_path : str
        File path for the output PNG.
    """
    times = np.asarray(times, dtype=float)
    energy_truth = np.asarray(energy_truth, dtype=float)
    energy_vanilla = np.asarray(energy_vanilla, dtype=float)
    t_periods = _times_to_periods(times)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Reference horizontal line at the initial (true) energy
    H0 = energy_truth[0]
    ax.axhline(H0, color="black", linestyle="-.", linewidth=1.0,
               alpha=0.55, label=f"Initial energy  H₀ = {H0:.4f} km²/s²")

    # Traces
    ax.plot(t_periods, energy_truth,
            color=GROUND_TRUTH, linestyle="-", linewidth=1.8,
            label="Ground Truth (RK8)", zorder=4)

    ax.plot(t_periods, energy_vanilla,
            color=VANILLA_NN, linestyle=":", linewidth=1.8,
            label="Vanilla NN", zorder=3)

    if energy_pinn is not None:
        energy_pinn = np.asarray(energy_pinn, dtype=float)
        ax.plot(t_periods, energy_pinn,
                color=PINN, linestyle="--", linewidth=1.8,
                label="PINN", zorder=5)

    ax.set_xlabel("Time (orbital periods)")
    ax.set_ylabel("Specific Orbital Energy (km²/s²)")
    ax.set_title("Hamiltonian Energy Conservation")
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.35, linestyle="--")

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plotting] Energy conservation saved -> {save_path}")


# ---------------------------------------------------------------------------
# 3.  plot_loss_convergence
# ---------------------------------------------------------------------------

def plot_loss_convergence(
    loss_history_vanilla: np.ndarray,
    loss_history_pinn_data: np.ndarray,
    loss_history_pinn_physics: np.ndarray,
    save_path: str,
) -> None:
    """Plot training loss curves for both networks side by side.

    Left panel  : total loss — Vanilla MSE vs PINN total loss.
    Right panel : PINN decomposition — data loss vs physics loss.

    Parameters
    ----------
    loss_history_vanilla : np.ndarray, shape (E,)
        Per-epoch total MSE loss for the Vanilla NN.
    loss_history_pinn_data : np.ndarray, shape (E,)
        Per-epoch data-fitting component of the PINN loss.
    loss_history_pinn_physics : np.ndarray, shape (E,)
        Per-epoch physics-residual component of the PINN loss.
    save_path : str
        File path for the output PNG.
    """
    loss_history_vanilla = np.asarray(loss_history_vanilla, dtype=float)
    loss_history_pinn_data = np.asarray(loss_history_pinn_data, dtype=float)
    loss_history_pinn_physics = np.asarray(loss_history_pinn_physics, dtype=float)

    pinn_total = loss_history_pinn_data + loss_history_pinn_physics
    epochs_v = np.arange(1, len(loss_history_vanilla) + 1)
    epochs_p = np.arange(1, len(loss_history_pinn_data) + 1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: total loss comparison ----------------------------------------
    ax_left.semilogy(epochs_v, loss_history_vanilla,
                     color=VANILLA_NN, linestyle="-", linewidth=1.8,
                     label="Vanilla NN (MSE)")
    ax_left.semilogy(epochs_p, pinn_total,
                     color=PINN, linestyle="--", linewidth=1.8,
                     label="PINN (total)")
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Loss (log scale)")
    ax_left.set_title("Total Loss Convergence")
    ax_left.legend(loc="upper right", framealpha=0.85)
    ax_left.grid(True, which="both", alpha=0.35, linestyle="--")

    # ---- Right: PINN component breakdown ------------------------------------
    ax_right.semilogy(epochs_p, loss_history_pinn_data,
                      color=GROUND_TRUTH, linestyle="-", linewidth=1.8,
                      label="PINN data loss")
    ax_right.semilogy(epochs_p, loss_history_pinn_physics,
                      color=PINN, linestyle="--", linewidth=1.8,
                      label="PINN physics loss")
    ax_right.semilogy(epochs_p, pinn_total,
                      color="grey", linestyle=":", linewidth=1.2, alpha=0.7,
                      label="PINN total")
    ax_right.set_xlabel("Epoch")
    ax_right.set_ylabel("Loss (log scale)")
    ax_right.set_title("PINN Loss Components")
    ax_right.legend(loc="upper right", framealpha=0.85)
    ax_right.grid(True, which="both", alpha=0.35, linestyle="--")

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plotting] Loss convergence saved -> {save_path}")


# ---------------------------------------------------------------------------
# 4.  plot_rmse_over_time
# ---------------------------------------------------------------------------

def plot_rmse_over_time(
    times: np.ndarray,
    errors_vanilla: np.ndarray,
    errors_pinn,
    save_path: str,
    train_fraction: float = 0.8,
) -> None:
    """Plot position error (km) over time for both networks.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time stamps in seconds.
    errors_vanilla : np.ndarray, shape (N,)
        Per-step position error for the Vanilla NN [km].
    errors_pinn : np.ndarray or None, shape (N,)
        Per-step position error for the PINN [km].  Pass ``None`` to skip.
    save_path : str
        File path for the output PNG.
    train_fraction : float
        Fraction of the time axis used for training, used to draw the
        train/test boundary line.  Default 0.7 (70 % train, 30 % test).
    """
    times = np.asarray(times, dtype=float)
    errors_vanilla = np.asarray(errors_vanilla, dtype=float)
    t_periods = _times_to_periods(times)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t_periods, errors_vanilla,
            color=VANILLA_NN, linestyle="-", linewidth=1.6,
            label="Vanilla NN error")

    if errors_pinn is not None:
        errors_pinn = np.asarray(errors_pinn, dtype=float)
        ax.plot(t_periods, errors_pinn,
                color=PINN, linestyle="--", linewidth=1.6,
                label="PINN error")

    # Train / test boundary
    t_boundary = t_periods[int(len(t_periods) * train_fraction)]
    ax.axvline(x=t_boundary, color="black", linestyle="--", linewidth=1.2,
               alpha=0.70, label=f"Train / Test boundary ({train_fraction:.0%})")

    # Region annotations
    y_top = ax.get_ylim()[1]
    ax.text(t_boundary * 0.5, y_top * 0.92, "Train",
            ha="center", va="top", fontsize=10,
            color="black", alpha=0.55)
    ax.text(t_boundary + (t_periods[-1] - t_boundary) * 0.5, y_top * 0.92,
            "Test", ha="center", va="top", fontsize=10,
            color="black", alpha=0.55)

    ax.set_xlabel("Time (orbital periods)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title("Position Error Over Time")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plotting] RMSE over time saved -> {save_path}")


# ---------------------------------------------------------------------------
# 5.  plot_error_distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(
    errors_vanilla: np.ndarray,
    errors_pinn,
    save_path: str,
    n_bins: int = 60,
) -> None:
    """Side-by-side histograms of position errors with statistical annotations.

    Parameters
    ----------
    errors_vanilla : np.ndarray, shape (N,)
        Position errors for the Vanilla NN [km].
    errors_pinn : np.ndarray or None, shape (N,)
        Position errors for the PINN [km].  Pass ``None`` to show only Vanilla.
    save_path : str
        File path for the output PNG.
    n_bins : int
        Number of histogram bins.
    """
    errors_vanilla = np.asarray(errors_vanilla, dtype=float)

    n_panels = 2 if errors_pinn is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    # ---- Helper to draw one histogram panel ---------------------------------
    def _draw_panel(ax, errors, color, label):
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        ax.hist(errors, bins=n_bins, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.axvline(mean_err, color="black", linestyle="--", linewidth=1.4,
                   label=f"Mean = {mean_err:.3f} km")
        # Annotation box
        ax.text(
            0.97, 0.95,
            f"Mean:  {mean_err:.3f} km\nMax:   {max_err:.3f} km",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, alpha=0.85),
        )
        ax.set_xlabel("Position Error (km)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\nError Distribution")
        ax.legend(loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.35, linestyle="--")

    _draw_panel(axes[0], errors_vanilla, VANILLA_NN, "Vanilla NN")

    if errors_pinn is not None:
        errors_pinn = np.asarray(errors_pinn, dtype=float)
        _draw_panel(axes[1], errors_pinn, PINN, "PINN")

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plotting] Error distribution saved -> {save_path}")


# ---------------------------------------------------------------------------
# 6.  create_rmse_table
# ---------------------------------------------------------------------------

def create_rmse_table(
    times: np.ndarray,
    errors_vanilla: np.ndarray,
    errors_pinn,
    output_path: str,
    checkpoints: tuple = (100, 1000, 5000, 10000),
) -> None:
    """Compute and display RMSE at specific time checkpoints.

    A formatted table is both printed to stdout and saved as a plain-text
    file at ``output_path``.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time stamps in seconds.
    errors_vanilla : np.ndarray, shape (N,)
        Per-step position errors for the Vanilla NN [km].
    errors_pinn : np.ndarray or None, shape (N,)
        Per-step position errors for the PINN [km].  ``None`` allowed.
    output_path : str
        Path for the text file output.
    checkpoints : tuple of float
        Time instants (seconds) at which to evaluate point RMSE.
    """
    times = np.asarray(times, dtype=float)
    errors_vanilla = np.asarray(errors_vanilla, dtype=float)
    has_pinn = errors_pinn is not None
    if has_pinn:
        errors_pinn = np.asarray(errors_pinn, dtype=float)

    # ---- Table header -------------------------------------------------------
    col_w = 22
    header = (
        f"{'Time (s)':>{col_w}} | "
        f"{'Vanilla NN RMSE (km)':>{col_w}} | "
        f"{'PINN RMSE (km)':>{col_w}} | "
        f"{'Improvement (%)':>{col_w}}"
    )
    sep = "-" * len(header)
    rows = [
        "=" * len(header),
        "  RMSE Checkpoint Table — ISEF PINN Orbital Propagation",
        "=" * len(header),
        header,
        sep,
    ]

    for t_ck in checkpoints:
        # Find the index closest to the requested checkpoint time
        idx = int(np.argmin(np.abs(times - t_ck)))
        t_actual = times[idx]

        # Compute cumulative RMSE up to this checkpoint
        rmse_v = float(np.sqrt(np.mean(errors_vanilla[: idx + 1] ** 2)))
        rmse_p = float(np.sqrt(np.mean(errors_pinn[: idx + 1] ** 2))) \
            if has_pinn else float("nan")

        if has_pinn and rmse_v > 0:
            improvement = (rmse_v - rmse_p) / rmse_v * 100.0
        else:
            improvement = float("nan")

        rmse_p_str = f"{rmse_p:.6f}" if has_pinn else "N/A"
        impr_str = f"{improvement:.2f}%" if has_pinn else "N/A"

        row = (
            f"{t_actual:>{col_w}.1f} | "
            f"{rmse_v:>{col_w}.6f} | "
            f"{rmse_p_str:>{col_w}} | "
            f"{impr_str:>{col_w}}"
        )
        rows.append(row)

    rows.append("=" * len(header))
    table_str = "\n".join(rows)

    # Print
    print(table_str)

    # Save
    _ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(table_str + "\n")
    print(f"[plotting] RMSE table saved -> {output_path}")


# ---------------------------------------------------------------------------
# 7.  plot_2d_trajectory_comparison
# ---------------------------------------------------------------------------

def plot_2d_trajectory_comparison(
    ground_truth: np.ndarray,
    vanilla_pred: np.ndarray,
    pinn_pred,
    save_path: str,
    train_fraction: float = 0.8,
) -> None:
    """Top-down (X–Y) projection of the orbital trajectory comparison.

    Provides a cleaner, two-dimensional view that is easy to read on a poster.

    Parameters
    ----------
    ground_truth : np.ndarray, shape (N, 2+)
        Ground-truth positions; first two columns are X and Y in km.
    vanilla_pred : np.ndarray or None, shape (N, 2+)
        Vanilla NN predicted positions.
    pinn_pred : np.ndarray or None, shape (N, 2+)
        PINN predicted positions.  Pass ``None`` to skip.
    save_path : str
        File path for the output PNG.
    train_fraction : float
        Fraction of trajectory used for training.  Used to mark the
        train / test boundary point on the orbit.
    """
    ground_truth = np.asarray(ground_truth)
    gt_x, gt_y = ground_truth[:, 0], ground_truth[:, 1]

    fig, ax = plt.subplots(figsize=(9, 9))

    # --- Trajectories --------------------------------------------------------
    ax.plot(gt_x, gt_y,
            color=GROUND_TRUTH, linestyle="-", linewidth=1.8,
            label="Ground Truth (RK8)", zorder=4)

    if vanilla_pred is not None:
        vanilla_pred = np.asarray(vanilla_pred)
        ax.plot(vanilla_pred[:, 0], vanilla_pred[:, 1],
                color=VANILLA_NN, linestyle=":", linewidth=1.8,
                label="Vanilla NN", zorder=3)

    if pinn_pred is not None:
        pinn_pred = np.asarray(pinn_pred)
        ax.plot(pinn_pred[:, 0], pinn_pred[:, 1],
                color=PINN, linestyle="--", linewidth=1.8,
                label="PINN", zorder=5)

    # --- Earth circle --------------------------------------------------------
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(R_EARTH * np.cos(theta), R_EARTH * np.sin(theta),
            color=EARTH, linewidth=1.2, alpha=0.6, label="Earth surface")
    ax.fill(R_EARTH * np.cos(theta), R_EARTH * np.sin(theta),
            color=EARTH, alpha=0.12)

    # --- Start / end markers on ground truth ---------------------------------
    ax.scatter([gt_x[0]], [gt_y[0]],
               color=GROUND_TRUTH, marker="o", s=70, zorder=7, label="Start")
    ax.scatter([gt_x[-1]], [gt_y[-1]],
               color=GROUND_TRUTH, marker="s", s=70, zorder=7, label="End")

    # --- Train / test boundary -----------------------------------------------
    boundary_idx = int(len(gt_x) * train_fraction)
    ax.scatter([gt_x[boundary_idx]], [gt_y[boundary_idx]],
               color="black", marker="D", s=55, zorder=8,
               label=f"Train/Test boundary ({train_fraction:.0%})")
    ax.annotate(
        "Train | Test",
        xy=(gt_x[boundary_idx], gt_y[boundary_idx]),
        xytext=(gt_x[boundary_idx] + 300, gt_y[boundary_idx] + 300),
        fontsize=9, color="black", alpha=0.70,
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.55),
    )

    # --- Axes ----------------------------------------------------------------
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("2-D Trajectory Comparison (X–Y Projection)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.35, linestyle="--")

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plotting] 2D trajectory comparison saved -> {save_path}")


# ---------------------------------------------------------------------------
# Smoke-test / demo  (run this file directly to verify everything imports)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("  plotting.py — self-test with synthetic data")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Generate a synthetic circular orbit as synthetic ground truth
    # ------------------------------------------------------------------
    T_DEMO = _T_ORBIT           # one full orbital period
    N = 500
    t_demo = np.linspace(0, T_DEMO, N)
    omega = 2 * math.pi / T_DEMO

    gt = np.column_stack([
        R_ORBIT * np.cos(omega * t_demo),
        R_ORBIT * np.sin(omega * t_demo),
        np.zeros(N),
    ])

    # Vanilla NN: add growing positional noise
    rng = np.random.default_rng(0)
    drift = np.linspace(0, 200, N)
    vanilla = gt + rng.normal(0, 1, gt.shape) * drift[:, None]

    # PINN: add small bounded noise
    pinn = gt + rng.normal(0, 1, gt.shape) * 5.0

    # ------------------------------------------------------------------
    # Synthetic energy arrays
    # ------------------------------------------------------------------
    v_circ = math.sqrt(MU / R_ORBIT)
    H0 = 0.5 * v_circ ** 2 - MU / R_ORBIT
    energy_truth = np.full(N, H0) + rng.normal(0, 1e-6, N)
    energy_vanilla = H0 + np.linspace(0, 50, N) + rng.normal(0, 2, N)
    energy_pinn = np.full(N, H0) + rng.normal(0, 0.05, N)

    # ------------------------------------------------------------------
    # Synthetic losses
    # ------------------------------------------------------------------
    E = 200
    epochs = np.arange(1, E + 1)
    loss_vanilla = 1.0 * np.exp(-epochs / 40) + rng.uniform(0, 0.01, E)
    loss_pinn_data = 0.8 * np.exp(-epochs / 45) + rng.uniform(0, 0.005, E)
    loss_pinn_phys = 0.5 * np.exp(-epochs / 60) + rng.uniform(0, 0.003, E)

    # ------------------------------------------------------------------
    # Synthetic errors
    # ------------------------------------------------------------------
    errors_v = np.linalg.norm(vanilla - gt, axis=1)
    errors_p = np.linalg.norm(pinn - gt, axis=1)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = os.path.join(_PROJECT_ROOT, "figures", "test")

    plot_3d_comparison(gt, vanilla, pinn,
                       save_path=os.path.join(out_dir, "test_3d.png"))

    plot_energy_conservation(t_demo, energy_truth, energy_vanilla, energy_pinn,
                             save_path=os.path.join(out_dir, "test_energy.png"))

    plot_loss_convergence(loss_vanilla, loss_pinn_data, loss_pinn_phys,
                          save_path=os.path.join(out_dir, "test_loss.png"))

    plot_rmse_over_time(t_demo, errors_v, errors_p,
                        save_path=os.path.join(out_dir, "test_rmse.png"))

    plot_error_distribution(errors_v, errors_p,
                            save_path=os.path.join(out_dir, "test_hist.png"))

    create_rmse_table(t_demo, errors_v, errors_p,
                      output_path=os.path.join(out_dir, "test_rmse_table.txt"))

    plot_2d_trajectory_comparison(gt, vanilla, pinn,
                                  save_path=os.path.join(out_dir, "test_2d.png"))

    print("\n[plotting] All self-test figures saved to", out_dir)
    print("=" * 60)
