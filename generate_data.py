"""
generate_data.py — Ground-truth orbital data generation for the ISEF PINN project.

Produces high-accuracy numerical solutions of the two-body and J2-perturbed
orbital equations of motion using SciPy's DOP853 integrator.  The resulting
.npy arrays are consumed downstream by the PINN training and evaluation
pipelines, so the column layout **must** remain:

    [t, x, y, z, vx, vy, vz]          (shape: n_points x 7)

Units: km and km/s throughout (SI-gravitational with km length scale).
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

# Use the non-interactive Agg backend so the script works headlessly.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3-D projection)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `src.physics` resolves
# regardless of the working directory from which this script is invoked.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.physics import (
    MU,
    R_EARTH,
    J2,
    R_ORBIT,
    circular_velocity,
    orbital_period,
    initial_state_vector,
    two_body_ode,
    j2_perturbation_ode,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_orbital_data(n_periods=3, n_points=2000, use_j2=False):
    """Integrate the equations of motion and return the trajectory.

    Parameters
    ----------
    n_periods : int or float
        Number of orbital periods to propagate.
    n_points : int
        Number of evenly-spaced output time steps.
    use_j2 : bool
        If *True*, include J2 zonal-harmonic perturbation; otherwise use
        the Keplerian two-body model.

    Returns
    -------
    data : np.ndarray, shape (n_points, 7)
        Columns are ``[t, x, y, z, vx, vy, vz]``.
    """
    y0 = initial_state_vector()
    T = orbital_period(R_ORBIT)
    t_span = (0.0, n_periods * T)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    ode_func = j2_perturbation_ode if use_j2 else two_body_ode

    print(f"Integrating {'J2-perturbed' if use_j2 else 'two-body'} ODE "
          f"for {n_periods} periods ({t_span[1]:.1f} s), "
          f"{n_points} output points ...")

    sol = solve_ivp(
        fun=ode_func,
        t_span=t_span,
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # sol.y has shape (6, n_points); transpose and prepend the time column.
    t_col = sol.t.reshape(-1, 1)          # (n_points, 1)
    state = sol.y.T                        # (n_points, 6)
    data = np.hstack([t_col, state])       # (n_points, 7)

    print(f"  -> Integration complete.  Solution shape: {data.shape}")
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_orbit(data):
    """Run basic sanity checks on an orbital trajectory.

    Parameters
    ----------
    data : np.ndarray, shape (N, 7)
        Columns ``[t, x, y, z, vx, vy, vz]``.
    """
    t = data[:, 0]
    pos = data[:, 1:4]   # x, y, z
    vel = data[:, 4:7]   # vx, vy, vz

    r = np.linalg.norm(pos, axis=1)
    v = np.linalg.norm(vel, axis=1)

    # --- Radial distance check ---
    max_dev = np.max(np.abs(r - R_ORBIT))
    mean_r = np.mean(r)
    print("\n--- Orbit validation ---")
    print(f"  Expected orbit radius : {R_ORBIT:.3f} km")
    print(f"  Mean radius           : {mean_r:.6f} km")
    print(f"  Max radial deviation  : {max_dev:.6e} km")

    # --- Specific orbital energy conservation ---
    # H = 0.5 * v^2 - mu / r
    energy = 0.5 * v**2 - MU / r
    H_first = energy[0]
    H_last = energy[-1]
    delta_H = abs(H_last - H_first)
    rel_delta_H = delta_H / abs(H_first) if H_first != 0.0 else delta_H

    print(f"  Energy (first point)  : {H_first:.10e} km^2/s^2")
    print(f"  Energy (last  point)  : {H_last:.10e} km^2/s^2")
    print(f"  |dH|                  : {delta_H:.6e} km^2/s^2")
    print(f"  |dH/H|               : {rel_delta_H:.6e}")
    print("--- End validation ---\n")


# ---------------------------------------------------------------------------
# 3-D plotting
# ---------------------------------------------------------------------------

def plot_orbit_3d(data, title="Orbit", save_path="orbit.png"):
    """Create a publication-quality 3-D orbit plot.

    Parameters
    ----------
    data : np.ndarray, shape (N, 7)
        Columns ``[t, x, y, z, vx, vy, vz]``.
    title : str
        Plot title.
    save_path : str
        File path for the saved PNG image.
    """
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Trajectory ---
    ax.plot(x, y, z, linewidth=0.6, color="royalblue", label="Trajectory")

    # --- Earth wireframe ---
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ex = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ey = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ez = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(ex, ey, ez, color="forestgreen", alpha=0.25,
                      linewidth=0.4, label="Earth")

    # --- Labels and formatting ---
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio for all three axes
    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min(),
    ) / 2.0
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate, validate, plot, and persist orbital datasets."""

    data_dir = os.path.join(_PROJECT_ROOT, "data")
    fig_dir = os.path.join(_PROJECT_ROOT, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Two-body (Keplerian) dataset ----
    print("=" * 60)
    print("  TWO-BODY (KEPLERIAN) DATA GENERATION")
    print("=" * 60)
    data_twobody = generate_orbital_data(
        n_periods=3, n_points=2000, use_j2=False
    )
    validate_orbit(data_twobody)

    # Save as both the default and the explicitly-named file.
    path_default = os.path.join(data_dir, "orbital_data.npy")
    path_twobody = os.path.join(data_dir, "orbital_data_twobody.npy")
    np.save(path_default, data_twobody)
    np.save(path_twobody, data_twobody)
    print(f"  Saved -> {path_default}")
    print(f"  Saved -> {path_twobody}")

    plot_orbit_3d(
        data_twobody,
        title="Two-Body Orbit (Ground Truth)",
        save_path=os.path.join(fig_dir, "orbit_ground_truth.png"),
    )

    # ---- J2-perturbed dataset ----
    print("=" * 60)
    print("  J2-PERTURBED DATA GENERATION")
    print("=" * 60)
    data_j2 = generate_orbital_data(
        n_periods=5, n_points=5000, use_j2=True
    )
    validate_orbit(data_j2)

    path_j2 = os.path.join(data_dir, "orbital_data_j2.npy")
    np.save(path_j2, data_j2)
    print(f"  Saved -> {path_j2}")

    plot_orbit_3d(
        data_j2,
        title="J2-Perturbed Orbit (Ground Truth)",
        save_path=os.path.join(fig_dir, "orbit_j2_ground_truth.png"),
    )

    # ---- Summary ----
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for label, d in [("Two-body", data_twobody), ("J2-perturbed", data_j2)]:
        t = d[:, 0]
        pos = d[:, 1:4]
        r = np.linalg.norm(pos, axis=1)
        print(f"\n  [{label}]")
        print(f"    Total propagation time : {t[-1]:.2f} s  "
              f"({t[-1] / 3600.0:.4f} h)")
        print(f"    Number of points       : {d.shape[0]}")
        print(f"    Orbit radius range     : "
              f"[{r.min():.6f}, {r.max():.6f}] km")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
