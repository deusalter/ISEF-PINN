"""
Shared neural network architectures for PINN orbital propagation.

Provides:
    - FourierPINN: Fourier-featured NN with secular drift head
    - VanillaMLP:  Plain tanh MLP baseline (no Fourier features)
    - CorrectionNetwork: Small NN for residual acceleration correction
    - NeuralODE: Neural ODE integrator for long-arc orbital propagation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics import gravity_j2j5_torch


# -- Default architecture constants ------------------------------------------

N_FREQ = 8       # Fourier frequencies
HIDDEN = 64      # Hidden layer width
LAYERS = 3       # Hidden layers


# -- FourierPINN --------------------------------------------------------------

class FourierPINN(nn.Module):
    """Fourier-featured neural network with secular drift for orbital prediction.

    Architecture:
        pos(t) = periodic_net(Fourier_enc(t))  +  sec_head(Fourier_enc(t)) * t

    The secular drift head captures slow monotonic drift of orbital elements
    (RAAN precession, argument-of-perigee precession) that manifests in ECI
    coordinates as t-modulated Fourier oscillations.

    The sec_head weights are zero-initialized so the model starts as a purely
    periodic network and only grows secular terms as training demands them.

    For long-arc propagation (multi-day), set ``t_max`` to the maximum
    normalized time value.  This adds two extra input features (t/t_max,
    (t/t_max)^2) that give the network a unique, monotonic signal to
    disambiguate different orbits whose Fourier encodings are identical.
    """

    def __init__(self, n_freq=N_FREQ, hidden=HIDDEN, n_layers=LAYERS,
                 dropout_p=0.0, custom_freqs=None, t_max=None):
        super().__init__()
        self.n_freq = n_freq
        self.dropout_p = dropout_p
        self._mc_dropout = False

        # Extra time features for long-arc disambiguation
        self._use_time_feats = t_max is not None
        n_time_feats = 2 if self._use_time_feats else 0
        if self._use_time_feats:
            self.register_buffer("_t_max",
                                 torch.tensor(float(t_max), dtype=torch.float64))

        input_dim = 2 * n_freq + n_time_feats
        if custom_freqs is not None:
            self.register_buffer(
                "freqs", torch.tensor(custom_freqs, dtype=torch.float64)
            )
        else:
            self.register_buffer(
                "freqs", torch.arange(1, n_freq + 1, dtype=torch.float64)
            )
        # Periodic backbone
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

        # Secular drift head: learns t-modulated Fourier coefficients.
        # Zero-initialized: starts as no drift, grows only when data requires it.
        self.sec_head = nn.Linear(input_dim, 3, bias=False)
        nn.init.zeros_(self.sec_head.weight)

        # Xavier init for all linear layers except sec_head
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.sec_head:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, t):
        wt = t * self.freqs
        parts = [torch.sin(wt), torch.cos(wt)]
        if self._use_time_feats:
            t_01 = t / self._t_max           # [0, 1]
            parts = [t_01, t_01 * t_01] + parts
        return torch.cat(parts, dim=1)

    def forward(self, t):
        enc = self.encode(t)                    # (N, 2*N_FREQ [+2])

        # MC Dropout path: apply F.dropout between layers (no nn.Dropout modules
        # so state dict keys remain identical to the original model).
        if self._mc_dropout and self.dropout_p > 0:
            x = enc
            for layer in self.net:
                x = layer(x)
                if isinstance(layer, nn.Tanh):
                    x = F.dropout(x, p=self.dropout_p, training=True)
            periodic = x
        else:
            periodic = self.net(enc)            # fast path -- no dropout

        secular = self.sec_head(enc) * t        # (N, 3) -- t-modulated drift part
        return periodic + secular

    # -- MC Dropout helpers ---------------------------------------------------

    def enable_dropout(self):
        """Enable MC Dropout for uncertainty estimation."""
        self._mc_dropout = True

    def disable_dropout(self):
        """Disable MC Dropout (restore fast deterministic path)."""
        self._mc_dropout = False

    def mc_forward(self, t, n_mc=50):
        """Run *n_mc* stochastic forward passes and return stacked results.

        Returns
        -------
        samples : Tensor, shape (n_mc, N, 3)
        """
        self.enable_dropout()
        samples = torch.stack([self.forward(t) for _ in range(n_mc)])
        self.disable_dropout()
        return samples


# -- VanillaMLP ---------------------------------------------------------------

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


# -- CorrectionNetwork -------------------------------------------------------

class CorrectionNetwork(nn.Module):
    """Small NN that learns residual accelerations (drag, SRP, third-body).

    Input:  6D normalized state [x/r_ref, y/r_ref, z/r_ref, vx/v_ref, vy/v_ref, vz/v_ref]
    Output: 3D acceleration correction (normalized, scaled by learnable gate)

    Architecture: 6 -> hidden -> hidden -> 3 (tanh activations)
    Final layer is zero-initialized so corrections start at zero.
    A learnable log-scale parameter controls the magnitude of the output.
    """

    def __init__(self, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(6, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

        # Xavier init for hidden layers, zero-init for output layer
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init final linear layer so corrections start at zero
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        # Learnable log-scale (starts at exp(-2) ≈ 0.14 for stronger
        # gradient signal; zero-init output layer still ensures correction
        # starts at zero, but weight gradients are amplified so the NN
        # can escape the near-zero regime faster)
        self.log_scale = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

    def forward(self, state_normalized: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_normalized : (N, 6) normalized [pos/r_ref, vel/v_ref]

        Returns
        -------
        (N, 3) acceleration correction (normalized units)
        """
        raw = self.net(state_normalized)       # (N, 3)
        return raw * torch.exp(self.log_scale)  # gated output


# -- NeuralODE ---------------------------------------------------------------

class NeuralODE(nn.Module):
    """Neural ODE for long-arc orbital propagation.

    Learns dynamics: dx/dt = f_known(x) + f_nn(x)
      - f_known: J2-J5 gravity (physical km/s²)
      - f_nn:    CorrectionNetwork residual (drag, SRP, third-body)

    Integration via manual RK4 (no external dependencies).
    All computations in physical units (km, km/s, seconds).
    NN input/output normalized internally using r_ref and v_ref.
    """

    def __init__(self, r_ref: float, v_ref: float,
                 hidden: int = 32, n_layers: int = 2):
        super().__init__()
        self.r_ref = r_ref
        self.v_ref = v_ref
        # Acceleration reference: v_ref / t_ref, but t_ref = r_ref / v_ref
        # so a_ref = v_ref^2 / r_ref
        self.a_ref = v_ref ** 2 / r_ref

        self.correction = CorrectionNetwork(hidden=hidden, n_layers=n_layers)

    def known_acceleration(self, pos_km: torch.Tensor) -> torch.Tensor:
        """J2-J5 gravitational acceleration in km/s².

        Parameters
        ----------
        pos_km : (N, 3) positions in km

        Returns
        -------
        (N, 3) acceleration in km/s²
        """
        return gravity_j2j5_torch(pos_km)

    def nn_acceleration(self, state_km: torch.Tensor) -> torch.Tensor:
        """NN correction acceleration in km/s².

        Parameters
        ----------
        state_km : (N, 6) [pos_km, vel_kms]

        Returns
        -------
        (N, 3) acceleration correction in km/s²
        """
        # Normalize state for NN input
        state_norm = torch.empty_like(state_km)
        state_norm[:, :3] = state_km[:, :3] / self.r_ref
        state_norm[:, 3:] = state_km[:, 3:] / self.v_ref
        # Get normalized correction, convert to physical
        a_norm = self.correction(state_norm)  # (N, 3)
        return a_norm * self.a_ref

    def dynamics(self, state_km: torch.Tensor) -> torch.Tensor:
        """Full RHS: d/dt [pos, vel] = [vel, a_gravity + a_nn].

        Parameters
        ----------
        state_km : (N, 6) [pos_km, vel_kms]

        Returns
        -------
        (N, 6) [vel_kms, accel_kms2]
        """
        pos = state_km[:, :3]
        vel = state_km[:, 3:]
        a_grav = self.known_acceleration(pos)
        a_nn = self.nn_acceleration(state_km)
        return torch.cat([vel, a_grav + a_nn], dim=1)

    def rk4_step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Single RK4 integration step.

        Parameters
        ----------
        state : (N, 6) current state in physical units
        dt : float, time step in seconds

        Returns
        -------
        (N, 6) state after one RK4 step
        """
        k1 = self.dynamics(state)
        k2 = self.dynamics(state + 0.5 * dt * k1)
        k3 = self.dynamics(state + 0.5 * dt * k2)
        k4 = self.dynamics(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_batched(
        self,
        states0: torch.Tensor,
        rel_times: torch.Tensor,
        dt: float = 60.0,
    ) -> torch.Tensor:
        """Integrate a batch of initial states simultaneously.

        Batched version of the original integrate_to_times: uses exact
        fractional RK4 steps at each eval time (preserving gradient flow
        through dynamics at every evaluation point).  All B trajectories
        are stepped forward in parallel using vectorized dynamics.

        Parameters
        ----------
        states0 : (B, 6) batch of initial states [km, km/s]
        rel_times : (M,) evaluation times relative to t=0 (seconds)
        dt : float, RK4 step size in seconds

        Returns
        -------
        (B, M, 6) predicted states at each eval time
        """
        eval_np = rel_times.detach().cpu().numpy()
        results = []
        current_states = states0  # (B, 6)
        current_t = 0.0

        for target_t in eval_np:
            # Full steps until close to target
            while current_t + dt < target_t:
                current_states = self.rk4_step(current_states, dt)
                current_t += dt

            # Fractional RK4 step to exact eval time
            remaining = target_t - current_t
            if remaining > 1e-6:
                final_states = self.rk4_step(current_states, remaining)
            else:
                final_states = current_states

            results.append(final_states)  # (B, 6)

        return torch.stack(results, dim=1)  # (B, M, 6)

    def integrate_batched_hermite(
        self,
        states0: torch.Tensor,
        rel_times: torch.Tensor,
        dt: float = 60.0,
    ) -> torch.Tensor:
        """Fast Hermite-interpolated batch integration (for inference).

        Uses full RK4 steps with Hermite cubic interpolation. Faster than
        integrate_batched but without per-eval-point gradient flow through
        dynamics. Best used under torch.no_grad() for predictions.

        Parameters
        ----------
        states0 : (B, 6) batch of initial states [km, km/s]
        rel_times : (M,) evaluation times relative to t=0 (seconds)
        dt : float, RK4 step size in seconds

        Returns
        -------
        (B, M, 6) predicted states at each eval time
        """
        t_end = rel_times[-1].item()
        n_steps = int(math.ceil(t_end / dt))

        # Full RK4 integration with derivatives for Hermite
        all_states = [states0]
        all_derivs = [self.dynamics(states0)]
        current = states0
        for _ in range(n_steps):
            current = self.rk4_step(current, dt)
            all_states.append(current)
            all_derivs.append(self.dynamics(current))
        all_states = torch.stack(all_states, dim=0)  # (n_steps+1, B, 6)
        all_derivs = torch.stack(all_derivs, dim=0)  # (n_steps+1, B, 6)

        # Hermite cubic interpolation
        step_idx = torch.clamp((rel_times / dt).long(), 0, n_steps - 1)
        tau = (rel_times - step_idx.to(rel_times.dtype) * dt) / dt
        tau = tau[:, None, None]  # (M, 1, 1)

        p0 = all_states[step_idx]
        p1 = all_states[step_idx + 1]
        m0 = all_derivs[step_idx] * dt
        m1 = all_derivs[step_idx + 1] * dt

        h00 = 2 * tau**3 - 3 * tau**2 + 1
        h10 = tau**3 - 2 * tau**2 + tau
        h01 = -2 * tau**3 + 3 * tau**2
        h11 = tau**3 - tau**2

        result = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        return result.permute(1, 0, 2)  # (B, M, 6)

    def integrate_to_times(
        self,
        state0: torch.Tensor,
        t0: float,
        eval_times: torch.Tensor,
        dt: float = 60.0,
    ) -> torch.Tensor:
        """Integrate from state0 at t0 to each time in eval_times.

        Uses RK4 with fixed step size dt and linear interpolation to
        exact eval_times (fast vectorized implementation).

        Parameters
        ----------
        state0 : (1, 6) or (6,) initial state [km, km/s]
        t0 : float, start time in seconds
        eval_times : (M,) times in seconds to evaluate at (must be >= t0)
        dt : float, RK4 step size in seconds

        Returns
        -------
        (M, 6) states at eval_times
        """
        if state0.dim() == 1:
            state0 = state0.unsqueeze(0)  # (1, 6)

        rel_times = eval_times - t0
        result = self.integrate_batched_hermite(state0, rel_times, dt=dt)
        return result.squeeze(0)  # (M, 6)
