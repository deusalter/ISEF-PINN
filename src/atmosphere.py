"""
Atmospheric density and drag perturbation module for PINN-based orbital
propagation.

Provides:
    - Harris-Priester atmospheric density model (PyTorch-differentiable)
    - Drag acceleration function (PyTorch-differentiable)

All computations use torch operations so that autograd can backpropagate
through the drag physics loss.

Units convention (same as physics.py):
    - Positions  : km
    - Velocities : km/s
    - Densities  : kg/m³
    - Accelerations returned : km/s²

Reference:
    Montenbruck & Gill, "Satellite Orbits", Springer 2000, Section 3.5
"""

import torch
import torch.nn as nn

from src.physics import R_EARTH

# ---------------------------------------------------------------------------
# Harris-Priester table (Montenbruck & Gill, Table 3.2)
# h in km, rho_min and rho_max in kg/m³
# ---------------------------------------------------------------------------

_HP_TABLE = [
    # (h_km,  rho_min,        rho_max)
    (100.0,  4.974e-07,  4.974e-07),
    (120.0,  2.490e-08,  2.490e-08),
    (130.0,  8.377e-09,  8.710e-09),
    (140.0,  3.899e-09,  4.059e-09),
    (150.0,  1.828e-09,  2.070e-09),
    (160.0,  9.550e-10,  1.065e-09),
    (170.0,  5.340e-10,  5.970e-10),
    (180.0,  3.070e-10,  3.540e-10),
    (190.0,  1.880e-10,  2.210e-10),
    (200.0,  1.170e-10,  1.440e-10),
    (210.0,  7.590e-11,  9.610e-11),
    (220.0,  5.010e-11,  6.530e-11),
    (230.0,  3.400e-11,  4.540e-11),
    (240.0,  2.350e-11,  3.230e-11),
    (250.0,  1.650e-11,  2.330e-11),
    (260.0,  1.190e-11,  1.720e-11),
    (270.0,  8.610e-12,  1.290e-11),
    (280.0,  6.340e-12,  9.710e-12),
    (290.0,  4.730e-12,  7.390e-12),
    (300.0,  3.560e-12,  5.700e-12),
    (320.0,  2.076e-12,  3.470e-12),
    (340.0,  1.233e-12,  2.160e-12),
    (360.0,  7.440e-13,  1.370e-12),
    (380.0,  4.580e-13,  8.790e-13),
    (400.0,  2.860e-13,  5.710e-13),
    (420.0,  1.810e-13,  3.790e-13),
    (440.0,  1.160e-13,  2.560e-13),
    (460.0,  7.570e-14,  1.760e-13),
    (480.0,  4.940e-14,  1.220e-13),
    (500.0,  3.290e-14,  8.590e-14),
    (520.0,  2.210e-14,  6.110e-14),
    (540.0,  1.500e-14,  4.390e-14),
    (560.0,  1.030e-14,  3.190e-14),
    (580.0,  7.200e-15,  2.340e-14),
    (600.0,  5.030e-15,  1.720e-14),
    (620.0,  3.570e-15,  1.290e-14),
    (640.0,  2.560e-15,  9.700e-15),
    (660.0,  1.840e-15,  7.350e-15),
    (680.0,  1.340e-15,  5.630e-15),
    (700.0,  9.820e-16,  4.330e-15),
    (720.0,  7.200e-16,  3.360e-15),
    (740.0,  5.370e-16,  2.640e-15),
    (760.0,  4.040e-16,  2.090e-15),
    (780.0,  3.050e-16,  1.670e-15),
    (800.0,  2.310e-16,  1.340e-15),
    (840.0,  1.340e-16,  8.760e-16),
    (900.0,  5.190e-17,  4.140e-16),
    (960.0,  2.040e-17,  1.990e-16),
    (1000.0, 1.040e-17,  1.170e-16),
]

# Pre-extract columns as plain Python lists (converted to tensors below)
_HP_H    = [row[0] for row in _HP_TABLE]
_HP_RMIN = [row[1] for row in _HP_TABLE]
_HP_RMAX = [row[2] for row in _HP_TABLE]

# Module-level tensors (float64 for numerical precision).
# Registered once; moved to the correct device by density functions on demand.
_HP_H_T    = torch.tensor(_HP_H,    dtype=torch.float64)   # (N_pts,)
_HP_LRMIN_T = torch.log(torch.tensor(_HP_RMIN, dtype=torch.float64))  # (N_pts,)
_HP_LRMAX_T = torch.log(torch.tensor(_HP_RMAX, dtype=torch.float64))  # (N_pts,)

_H_MIN = _HP_H[0]    # 100 km
_H_MAX = _HP_H[-1]   # 1000 km


# ---------------------------------------------------------------------------
# Density model
# ---------------------------------------------------------------------------

def harris_priester_density_torch(
    h_km: torch.Tensor,
    solar_mix: float = 0.5,
) -> torch.Tensor:
    """Harris-Priester atmospheric density model (differentiable, PyTorch).

    Parameters
    ----------
    h_km : torch.Tensor
        Altitude above Earth's surface in km.  Any shape is accepted; the
        result has the same shape.
    solar_mix : float, optional
        Solar activity interpolation factor in [0, 1].  0 = solar minimum
        (rho_min), 1 = solar maximum (rho_max), 0.5 = moderate (default).

    Returns
    -------
    torch.Tensor
        Atmospheric density in kg/m³, same shape as *h_km*.

    Notes
    -----
    Differentiability:  ``torch.bucketize`` returns integer indices which
    are not differentiable, but the *interpolation weight* computed from
    those indices IS differentiable w.r.t. h_km.  Autograd therefore works
    correctly through this function.
    """
    dev   = h_km.device
    dtype = h_km.dtype

    # Move table tensors to the same device / dtype as the input
    h_tbl    = _HP_H_T.to(device=dev, dtype=dtype)          # (N,)
    lrmin    = _HP_LRMIN_T.to(device=dev, dtype=dtype)      # (N,)
    lrmax    = _HP_LRMAX_T.to(device=dev, dtype=dtype)      # (N,)

    # Clamp altitude to valid table range [100, 1000] km
    h_clamped = torch.clamp(h_km, min=float(_H_MIN), max=float(_H_MAX))

    # Find the lower-bound table index for each altitude value.
    # bucketize returns index i such that h_tbl[i-1] <= h < h_tbl[i].
    # We want i_lo = i - 1 (lower bracket), clamped to [0, N-2].
    flat = h_clamped.reshape(-1)                       # (M,)
    n_pts = h_tbl.shape[0]

    # right=True means h_tbl[idx-1] <= h < h_tbl[idx]
    idx_hi = torch.bucketize(flat, h_tbl, right=True)         # (M,)  integer
    idx_hi = torch.clamp(idx_hi, min=1, max=n_pts - 1)        # keep in bounds
    idx_lo = idx_hi - 1                                        # (M,)  integer

    # Bracket altitudes and log-densities
    h_lo = h_tbl[idx_lo]    # (M,)
    h_hi = h_tbl[idx_hi]    # (M,)

    lrmin_lo = lrmin[idx_lo]
    lrmin_hi = lrmin[idx_hi]
    lrmax_lo = lrmax[idx_lo]
    lrmax_hi = lrmax[idx_hi]

    # Linear interpolation weight in altitude (differentiable w.r.t. flat)
    dh = h_hi - h_lo                                   # (M,) -- always > 0
    t  = (flat - h_lo) / dh                            # (M,) in [0, 1]

    # Log-linear interpolation of rho_min and rho_max separately
    log_rmin = lrmin_lo + t * (lrmin_hi - lrmin_lo)   # (M,)
    log_rmax = lrmax_lo + t * (lrmax_hi - lrmax_lo)   # (M,)

    # Blend between solar-min and solar-max
    log_rho = (1.0 - solar_mix) * log_rmin + solar_mix * log_rmax  # (M,)

    rho = torch.exp(log_rho)                           # (M,)
    return rho.reshape(h_km.shape)


# ---------------------------------------------------------------------------
# Drag acceleration
# ---------------------------------------------------------------------------

def drag_acceleration_torch(
    pos_km: torch.Tensor,
    vel_kms: torch.Tensor,
    cd_a_over_m: float | torch.Tensor,
    solar_mix: float = 0.5,
) -> torch.Tensor:
    """Compute atmospheric drag acceleration (differentiable, PyTorch).

    Parameters
    ----------
    pos_km : torch.Tensor, shape (N, 3)
        Satellite position in km (ECI frame, physical units).
    vel_kms : torch.Tensor, shape (N, 3)
        Satellite velocity in km/s (ECI frame, physical units).
    cd_a_over_m : float or torch.Tensor
        Ballistic coefficient Cd * A / m in m²/kg.
        A typical LEO cubesat value is ~0.022 m²/kg.
    solar_mix : float, optional
        Solar activity factor passed to the density model (default 0.5).

    Returns
    -------
    torch.Tensor, shape (N, 3)
        Drag acceleration in km/s².

    Notes
    -----
    The derivation of the unit conversion factor 1e3:

        a_drag [km/s²]
            = -0.5 * (Cd*A/m [m²/kg]) * (rho [kg/m³])
              * |v_rel [km/s]| * v_rel [km/s]  * C

        Dimensional check:
            [m²/kg] * [kg/m³] * [km/s] * [km/s]
            = [1/m] * [km²/s²]
            = [km²/(m·s²)]

        Converting km²/m to km/s²·km requires:
            km²/m = km² / (1e-3 km) = 1e3 km
        so the factor C = 1e3 makes units come out as km/s².

    The velocity of the co-rotating atmosphere is:
        v_atm = omega_E × pos_km
    where omega_E = [0, 0, 7.2921159e-5] rad/s.  Because pos is in km and
    omega in rad/s the cross product naturally yields km/s.
    """
    OMEGA_EARTH = 7.2921159e-5  # rad/s

    # Altitude of each point
    r_km = torch.norm(pos_km, dim=1, keepdim=True)      # (N, 1)  km
    h_km = r_km.squeeze(1) - R_EARTH                    # (N,)    km

    # Atmospheric density at each altitude
    rho = harris_priester_density_torch(h_km, solar_mix=solar_mix)  # (N,) kg/m³
    rho = rho.unsqueeze(1)                               # (N, 1)

    # Velocity of the co-rotating atmosphere: v_atm = omega × pos
    # omega = [0, 0, OMEGA_EARTH]
    # omega × [x, y, z] = [0*z - OMEGA_EARTH*y,  OMEGA_EARTH*x - 0*z,  0]
    #                    = [-OMEGA_EARTH*y,  OMEGA_EARTH*x,  0]   (km/s)
    px = pos_km[:, 0]   # (N,)
    py = pos_km[:, 1]   # (N,)
    v_atm = torch.stack([
        -OMEGA_EARTH * py,
         OMEGA_EARTH * px,
        torch.zeros_like(px),
    ], dim=1)                                             # (N, 3)  km/s

    # Velocity relative to rotating atmosphere
    v_rel = vel_kms - v_atm                              # (N, 3)  km/s

    # Speed (km/s)
    v_rel_mag = torch.norm(v_rel, dim=1, keepdim=True)   # (N, 1)  km/s

    # Drag acceleration (km/s²)
    # a = -0.5 * Cd*A/m * rho * |v_rel| * v_rel * 1e3
    a_drag = -0.5 * cd_a_over_m * rho * v_rel_mag * v_rel * 1.0e3  # (N, 3)

    return a_drag


# ---------------------------------------------------------------------------
# Verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Harris-Priester Atmosphere Module -- Verification")
    print("=" * 60)

    torch.set_default_dtype(torch.float64)

    # ------------------------------------------------------------------
    # 1.  Density sanity checks
    # ------------------------------------------------------------------
    h_test = torch.tensor([200.0, 400.0], dtype=torch.float64, requires_grad=True)
    rho_test = harris_priester_density_torch(h_test, solar_mix=0.5)

    print(f"\nDensity at 200 km : {rho_test[0].item():.4e} kg/m³")
    print(f"  (expected range: ~1e-10 to ~4e-10 kg/m³)")
    print(f"Density at 400 km : {rho_test[1].item():.4e} kg/m³")
    print(f"  (expected range: ~2e-13 to ~6e-13 kg/m³)")

    # ------------------------------------------------------------------
    # 2.  Differentiability of density
    # ------------------------------------------------------------------
    grad_rho = torch.autograd.grad(rho_test.sum(), h_test)[0]
    print(f"\nGradient d(rho)/d(h) at [200, 400] km:")
    print(f"  {grad_rho.tolist()}")
    print(f"  (both should be negative — density falls with altitude)")

    # ------------------------------------------------------------------
    # 3.  ISS-like drag acceleration
    # ------------------------------------------------------------------
    # ISS orbit: ~400 km circular, approx pos=[6778,0,0] km, vel=[0,7.67,0] km/s
    ISS_pos = torch.tensor([[6778.0, 0.0, 0.0]], dtype=torch.float64,
                            requires_grad=True)
    ISS_vel = torch.tensor([[0.0,    7.67, 0.0]], dtype=torch.float64)

    # Typical satellite: Cd*A/m ~ 0.022 m²/kg  (Cd=2.2, A=0.01 m², m=1 kg)
    Cd_A_over_m = 0.022

    a_drag = drag_acceleration_torch(ISS_pos, ISS_vel, Cd_A_over_m, solar_mix=0.5)
    a_mag  = torch.norm(a_drag, dim=1).item()

    print(f"\nISS-like drag acceleration:")
    print(f"  pos  = [6778, 0, 0] km    (h = {6778.0 - R_EARTH:.1f} km)")
    print(f"  vel  = [0, 7.67, 0] km/s")
    print(f"  Cd*A/m = {Cd_A_over_m} m²/kg")
    print(f"  a_drag = {a_drag.detach().squeeze().tolist()} km/s²")
    print(f"  |a_drag| = {a_mag:.4e} km/s²")
    print(f"  (expected range: ~1e-8 to ~1e-7 km/s²)")

    # ------------------------------------------------------------------
    # 4.  Differentiability of drag w.r.t. position
    # ------------------------------------------------------------------
    loss = a_drag.norm()
    loss.backward()
    print(f"\nGradient of |a_drag| w.r.t. pos:")
    print(f"  {ISS_pos.grad.squeeze().tolist()}")
    print(f"  (non-zero values confirm differentiability)")

    # ------------------------------------------------------------------
    # 5.  Density across full altitude range
    # ------------------------------------------------------------------
    h_range = torch.linspace(100.0, 1000.0, 10, dtype=torch.float64)
    rho_range = harris_priester_density_torch(h_range, solar_mix=0.5)
    print(f"\nDensity profile (solar_mix=0.5):")
    for hi, ri in zip(h_range.tolist(), rho_range.tolist()):
        print(f"  h={hi:6.0f} km  rho={ri:.4e} kg/m³")

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60)
