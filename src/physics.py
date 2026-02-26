"""
Physics foundation module for PINN-based orbital propagation.

Provides:
    - Astrodynamic constants (km / s units throughout)
    - Keplerian two-body equations of motion
    - J2-perturbed equations of motion
    - State normalization utilities
    - Hamiltonian (specific orbital energy) computation
    - Physics-residual functions for PINN loss (PyTorch)

All vectors follow the convention [x, y, z, vx, vy, vz].
"""

import math
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1.  Physical constants  (all in km and seconds)
# ---------------------------------------------------------------------------

MU: float = 398600.4418
"""Earth gravitational parameter  [km^3 / s^2]."""

R_EARTH: float = 6378.137
"""Earth equatorial radius  [km]."""

J2: float = 1.08263e-3
"""Earth oblateness (J2) coefficient  [dimensionless].  Used in Phase 4."""

J3: float = -2.53265e-6
"""Earth odd-harmonic (J3) coefficient  [dimensionless].
   Causes north-south asymmetry in gravity.  About 0.23% of J2."""

J4: float = -1.61990e-6
"""Earth even-harmonic (J4) coefficient  [dimensionless].
   Causes additional oblateness effects.  About 0.15% of J2."""

J5: float = -2.27271e-7
"""Earth odd-harmonic (J5) coefficient  [dimensionless].
   Higher-order north-south asymmetry.  About 0.02% of J2."""

LEO_ALTITUDE: float = 400.0
"""Reference LEO altitude above the equator  [km]."""

R_ORBIT: float = R_EARTH + LEO_ALTITUDE
"""Circular-orbit radius for the reference LEO  [km]."""


# ---------------------------------------------------------------------------
# 2.  circular_velocity
# ---------------------------------------------------------------------------

def circular_velocity(r: float) -> float:
    """Return the circular orbital velocity at radius *r*.

    Parameters
    ----------
    r : float
        Orbital radius  [km].

    Returns
    -------
    float
        Circular velocity  [km/s].  v = sqrt(MU / r).
    """
    return math.sqrt(MU / r)


# ---------------------------------------------------------------------------
# 3.  orbital_period
# ---------------------------------------------------------------------------

def orbital_period(r: float) -> float:
    """Return the Keplerian orbital period for a circular orbit at radius *r*.

    Parameters
    ----------
    r : float
        Orbital radius  [km].

    Returns
    -------
    float
        Period  [s].  T = 2 * pi * sqrt(r^3 / MU).
    """
    return 2.0 * math.pi * math.sqrt(r ** 3 / MU)


# ---------------------------------------------------------------------------
# 4.  initial_state_vector
# ---------------------------------------------------------------------------

def initial_state_vector() -> np.ndarray:
    """Return the canonical initial state for a 400 km circular LEO.

    The satellite starts at (R_ORBIT, 0, 0) with velocity (0, v_circ, 0),
    producing a prograde circular orbit in the xy-plane.

    Returns
    -------
    np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s.
    """
    v_circ = circular_velocity(R_ORBIT)
    return np.array([R_ORBIT, 0.0, 0.0, 0.0, v_circ, 0.0])


# ---------------------------------------------------------------------------
# 5.  two_body_ode
# ---------------------------------------------------------------------------

def two_body_ode(t: float, state: np.ndarray) -> np.ndarray:
    """Right-hand side of the Keplerian two-body ODE.

    Suitable for use with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    t : float
        Current time  [s]  (unused, but required by the integrator API).
    state : array_like, shape (6,)
        [x, y, z, vx, vy, vz].

    Returns
    -------
    np.ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az].
    """
    x, y, z, vx, vy, vz = state
    r = math.sqrt(x * x + y * y + z * z)
    r3 = r ** 3
    ax = -MU * x / r3
    ay = -MU * y / r3
    az = -MU * z / r3
    return np.array([vx, vy, vz, ax, ay, az])


# ---------------------------------------------------------------------------
# 6.  j2_perturbation_ode
# ---------------------------------------------------------------------------

def j2_perturbation_ode(t: float, state: np.ndarray) -> np.ndarray:
    """Right-hand side including the J2 oblateness perturbation.

    Computes the two-body acceleration *plus* the J2 perturbation.
    Suitable for use with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    t : float
        Current time  [s].
    state : array_like, shape (6,)
        [x, y, z, vx, vy, vz].

    Returns
    -------
    np.ndarray, shape (6,)
        [vx, vy, vz, ax_total, ay_total, az_total].
    """
    x, y, z, vx, vy, vz = state
    r = math.sqrt(x * x + y * y + z * z)
    r3 = r ** 3
    r5 = r ** 5

    # Two-body acceleration
    ax_tb = -MU * x / r3
    ay_tb = -MU * y / r3
    az_tb = -MU * z / r3

    # J2 perturbation acceleration
    z_term = (z / r) ** 2
    factor = -1.5 * J2 * MU * R_EARTH ** 2 / r5

    ax_j2 = factor * x * (1.0 - 5.0 * z_term)
    ay_j2 = factor * y * (1.0 - 5.0 * z_term)
    az_j2 = factor * z * (3.0 - 5.0 * z_term)

    return np.array([
        vx, vy, vz,
        ax_tb + ax_j2,
        ay_tb + ay_j2,
        az_tb + az_j2,
    ])


# ---------------------------------------------------------------------------
# 7.  NormalizationParams
# ---------------------------------------------------------------------------

class NormalizationParams:
    """Store and apply reference-value normalization for the orbital state.

    All quantities are scaled so that nominal circular-orbit values map to
    approximately 1.0, keeping the PINN inputs and outputs in a
    well-conditioned range.

    Attributes
    ----------
    r_ref : float
        Reference radius  [km].
    v_ref : float
        Reference velocity  [km/s].
    t_ref : float
        Reference time  [s].  Defined as T / (2*pi) so that one orbit
        corresponds to a normalized time of 2*pi.
    """

    def __init__(
        self,
        r_ref: float = R_ORBIT,
        v_ref: float | None = None,
        t_ref: float | None = None,
    ):
        self.r_ref = r_ref
        self.v_ref = v_ref if v_ref is not None else circular_velocity(r_ref)
        self.t_ref = t_ref if t_ref is not None else orbital_period(r_ref) / (2.0 * math.pi)

    # -- state ---------------------------------------------------------------

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize a 6-element state vector to approximately [-1, 1].

        Parameters
        ----------
        state : np.ndarray, shape (..., 6)
            Physical state [x, y, z, vx, vy, vz].

        Returns
        -------
        np.ndarray
            Normalized state.
        """
        state = np.asarray(state, dtype=np.float64)
        norm = np.empty_like(state)
        norm[..., :3] = state[..., :3] / self.r_ref
        norm[..., 3:] = state[..., 3:] / self.v_ref
        return norm

    def denormalize_state(self, norm_state: np.ndarray) -> np.ndarray:
        """Convert a normalized state back to physical units.

        Parameters
        ----------
        norm_state : np.ndarray, shape (..., 6)
            Normalized state.

        Returns
        -------
        np.ndarray
            Physical state [km, km, km, km/s, km/s, km/s].
        """
        norm_state = np.asarray(norm_state, dtype=np.float64)
        state = np.empty_like(norm_state)
        state[..., :3] = norm_state[..., :3] * self.r_ref
        state[..., 3:] = norm_state[..., 3:] * self.v_ref
        return state

    # -- time ----------------------------------------------------------------

    def normalize_time(self, t: float | np.ndarray) -> float | np.ndarray:
        """Normalize physical time to dimensionless time.

        Parameters
        ----------
        t : float or np.ndarray
            Time in seconds.

        Returns
        -------
        float or np.ndarray
            Normalized (dimensionless) time.
        """
        return t / self.t_ref

    def denormalize_time(self, t_norm: float | np.ndarray) -> float | np.ndarray:
        """Convert normalized time back to physical seconds.

        Parameters
        ----------
        t_norm : float or np.ndarray
            Normalized time.

        Returns
        -------
        float or np.ndarray
            Time in seconds.
        """
        return t_norm * self.t_ref

    def __repr__(self) -> str:
        return (
            f"NormalizationParams(r_ref={self.r_ref:.3f} km, "
            f"v_ref={self.v_ref:.4f} km/s, "
            f"t_ref={self.t_ref:.2f} s)"
        )


# ---------------------------------------------------------------------------
# 8.  hamiltonian_energy
# ---------------------------------------------------------------------------

def hamiltonian_energy(state: np.ndarray) -> float:
    """Compute the specific orbital energy (Hamiltonian) of a state.

    For a Keplerian orbit this quantity is conserved.

    Parameters
    ----------
    state : array_like, shape (6,)
        [x, y, z, vx, vy, vz].

    Returns
    -------
    float
        H = 0.5 * v^2 - MU / r   [km^2 / s^2].
    """
    x, y, z, vx, vy, vz = state
    r = math.sqrt(x * x + y * y + z * z)
    v_sq = vx * vx + vy * vy + vz * vz
    return 0.5 * v_sq - MU / r


# ---------------------------------------------------------------------------
# 9.  physics_residual_twobody  (PyTorch)
# ---------------------------------------------------------------------------

def physics_residual_twobody(
    pos: torch.Tensor,
    acc: torch.Tensor,
) -> torch.Tensor:
    """Compute the two-body physics residual for a batch of predictions.

    The residual is defined as

        f = acc + MU * pos / ||pos||^3

    and should be zero if the predicted acceleration exactly satisfies the
    two-body equation of motion.

    Parameters
    ----------
    pos : torch.Tensor, shape (N, 3)
        Predicted positions  [km].
    acc : torch.Tensor, shape (N, 3)
        Predicted accelerations  [km/s^2].

    Returns
    -------
    torch.Tensor, shape (N, 3)
        Physics residual.
    """
    r = torch.norm(pos, dim=1, keepdim=True)          # (N, 1)
    r3 = r ** 3                                        # (N, 1)
    residual = acc + MU * pos / r3                     # (N, 3)
    return residual


# ---------------------------------------------------------------------------
# 10. physics_residual_j2  (PyTorch)
# ---------------------------------------------------------------------------

def physics_residual_j2(
    pos: torch.Tensor,
    acc: torch.Tensor,
) -> torch.Tensor:
    """Compute the physics residual including J2 perturbation.

    The expected total acceleration is:

        a_expected = -MU * pos / r^3   +   a_J2

    where a_J2 is the J2 perturbation.  The residual is

        f = acc - a_expected

    and should be zero when the PINN prediction is physically consistent.

    Parameters
    ----------
    pos : torch.Tensor, shape (N, 3)
        Predicted positions  [km].
    acc : torch.Tensor, shape (N, 3)
        Predicted accelerations  [km/s^2].

    Returns
    -------
    torch.Tensor, shape (N, 3)
        Physics residual.
    """
    x = pos[:, 0:1]    # (N, 1)
    y = pos[:, 1:2]    # (N, 1)
    z = pos[:, 2:3]    # (N, 1)

    r = torch.norm(pos, dim=1, keepdim=True)           # (N, 1)
    r3 = r ** 3
    r5 = r ** 5

    # Two-body expected acceleration
    a_tb = -MU * pos / r3                               # (N, 3)

    # J2 perturbation expected acceleration
    z_term = (z / r) ** 2                                # (N, 1)
    factor = -1.5 * J2 * MU * R_EARTH ** 2 / r5         # (N, 1)

    a_j2_x = factor * x * (1.0 - 5.0 * z_term)         # (N, 1)
    a_j2_y = factor * y * (1.0 - 5.0 * z_term)         # (N, 1)
    a_j2_z = factor * z * (3.0 - 5.0 * z_term)         # (N, 1)

    a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)  # (N, 3)

    # Total expected acceleration
    a_expected = a_tb + a_j2                              # (N, 3)

    # Residual: should be zero if the prediction matches physics
    residual = acc - a_expected                           # (N, 3)
    return residual


# ---------------------------------------------------------------------------
# Verification (main block)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  PINN Orbital Propagation -- Physics Module Verification")
    print("=" * 60)

    # Circular velocity at 400 km LEO
    v_circ = circular_velocity(R_ORBIT)
    print(f"\nCircular velocity at {LEO_ALTITUDE:.0f} km altitude:")
    print(f"  r = R_EARTH + {LEO_ALTITUDE:.0f} km = {R_ORBIT:.3f} km")
    print(f"  v = sqrt(MU / r) = {v_circ:.4f} km/s")

    # Orbital period
    T = orbital_period(R_ORBIT)
    print(f"\nOrbital period at {LEO_ALTITUDE:.0f} km altitude:")
    print(f"  T = {T:.2f} s  ({T / 60.0:.2f} min)")

    # Initial state vector
    state0 = initial_state_vector()
    print(f"\nInitial state vector:")
    print(f"  pos = [{state0[0]:.3f}, {state0[1]:.3f}, {state0[2]:.3f}] km")
    print(f"  vel = [{state0[3]:.4f}, {state0[4]:.4f}, {state0[5]:.4f}] km/s")

    # Specific orbital energy (Hamiltonian)
    H = hamiltonian_energy(state0)
    print(f"\nSpecific orbital energy (Hamiltonian):")
    print(f"  H = {H:.4f} km^2/s^2")
    print(f"  (negative => bound orbit)")

    # Normalization sanity check
    norm = NormalizationParams()
    print(f"\nNormalization parameters:")
    print(f"  {norm}")
    ns = norm.normalize_state(state0)
    print(f"  Normalized state: {ns}")
    recovered = norm.denormalize_state(ns)
    print(f"  Round-trip error: {np.max(np.abs(recovered - state0)):.2e}")

    # Quick ODE check: derivative at initial state
    deriv = two_body_ode(0.0, state0)
    print(f"\nTwo-body derivative at t=0:")
    print(f"  d/dt = {deriv}")

    deriv_j2 = j2_perturbation_ode(0.0, state0)
    print(f"\nJ2-perturbed derivative at t=0:")
    print(f"  d/dt = {deriv_j2}")

    # Torch residual check
    pos_t = torch.tensor([[R_ORBIT, 0.0, 0.0]], dtype=torch.float64)
    acc_tb = torch.tensor([[deriv[3], deriv[4], deriv[5]]], dtype=torch.float64)
    res_tb = physics_residual_twobody(pos_t, acc_tb)
    print(f"\nTwo-body residual (should be ~0):")
    print(f"  {res_tb}")

    acc_j2 = torch.tensor([[deriv_j2[3], deriv_j2[4], deriv_j2[5]]], dtype=torch.float64)
    res_j2 = physics_residual_j2(pos_t, acc_j2)
    print(f"\nJ2-perturbed residual (should be ~0):")
    print(f"  {res_j2}")

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60)
