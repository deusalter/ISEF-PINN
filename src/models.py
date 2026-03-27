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

        # Learnable log-scale (starts small: exp(-6) ≈ 0.0025)
        self.log_scale = nn.Parameter(torch.tensor(-6.0, dtype=torch.float64))

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


# -- ConditionedCorrectionNetwork -------------------------------------------

class ConditionedCorrectionNetwork(nn.Module):
    """Correction network conditioned on orbital parameters and satellite embedding.

    Input: 14D = state(6) + phys_params(4) + embedding(4)
    Output: 3D acceleration correction (normalized, scaled by learnable gate)

    Physical params: [a_km, inc_deg, ecc, cd_a_over_m] (standardized internally).
    Embedding: per-satellite learned vector from nn.Embedding.
    Embedding dropout (20%): randomly zeros embedding during training, forcing
    the physical-params pathway to carry useful information for generalization.
    """

    def __init__(self, n_satellites: int = 21, embed_dim: int = 4,
                 hidden: int = 64, n_layers: int = 2,
                 embed_dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout

        # Satellite embedding (index 0 = unseen, initialized to zero)
        self.embedding = nn.Embedding(n_satellites, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.embedding.weight[0].zero_()  # index 0 = unseen

        # Physical param normalization constants (computed from catalog)
        # [a_km, inc_deg, ecc, cd_a_over_m]
        self.register_buffer("param_mean", torch.tensor(
            [6986.29, 58.04, 0.0008, 0.0216], dtype=torch.float64))
        self.register_buffer("param_std", torch.tensor(
            [150.16, 24.44, 0.0021, 0.0070], dtype=torch.float64))

        # MLP: 14 -> hidden -> hidden -> 3
        input_dim = 6 + 4 + embed_dim  # state + phys_params + embedding
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

        # Xavier init for hidden layers, zero-init for output layer
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        # Learnable log-scale (starts small: exp(-6) ~ 0.0025)
        self.log_scale = nn.Parameter(torch.tensor(-6.0, dtype=torch.float64))

    def forward(self, state_normalized: torch.Tensor,
                phys_params: torch.Tensor,
                sat_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_normalized : (N, 6) normalized [pos/r_ref, vel/v_ref]
        phys_params : (N, 4) raw [a_km, inc_deg, ecc, cd_a_over_m]
        sat_idx : (N,) integer satellite indices

        Returns
        -------
        (N, 3) acceleration correction (normalized units)
        """
        # Standardize physical params
        pp = (phys_params - self.param_mean) / self.param_std  # (N, 4)

        # Satellite embedding with inverted dropout
        # Scale surviving embeddings by 1/(1-p) so expected magnitude matches inference
        emb = self.embedding(sat_idx)  # (N, embed_dim)
        if self.training and self.embed_dropout > 0:
            mask = (torch.rand(emb.shape[0], 1, device=emb.device,
                               dtype=emb.dtype) > self.embed_dropout).to(emb.dtype)
            emb = emb * mask / (1.0 - self.embed_dropout)

        # Concatenate: [state(6), phys_params(4), embedding(4)] = 14D
        x = torch.cat([state_normalized, pp, emb], dim=1)
        raw = self.net(x)  # (N, 3)
        return raw * torch.exp(self.log_scale)


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
                current_states = self.rk4_step(current_states, remaining)
                current_t = target_t

            results.append(current_states)  # (B, 6)

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


# -- UniversalNeuralODE -----------------------------------------------------

class UniversalNeuralODE(nn.Module):
    """Universal Neural ODE for multi-satellite orbital propagation.

    Like NeuralODE but r_ref/v_ref are runtime arguments (not fixed attributes),
    allowing one model to propagate any satellite. The ConditionedCorrectionNetwork
    receives per-satellite physical parameters and a learned embedding.

    Integration via manual RK4 (no external dependencies).
    All computations in physical units (km, km/s, seconds).
    """

    def __init__(self, n_satellites: int = 21, embed_dim: int = 4,
                 hidden: int = 64, n_layers: int = 2,
                 embed_dropout: float = 0.2):
        super().__init__()
        self.correction = ConditionedCorrectionNetwork(
            n_satellites=n_satellites, embed_dim=embed_dim,
            hidden=hidden, n_layers=n_layers,
            embed_dropout=embed_dropout,
        )

    def nn_acceleration(self, state_km: torch.Tensor,
                        r_refs: torch.Tensor, v_refs: torch.Tensor,
                        phys_params: torch.Tensor,
                        sat_idxs: torch.Tensor) -> torch.Tensor:
        """NN correction acceleration in km/s^2.

        Parameters
        ----------
        state_km : (B, 6) [pos_km, vel_kms]
        r_refs : (B,) per-satellite reference radii [km]
        v_refs : (B,) per-satellite reference velocities [km/s]
        phys_params : (B, 4) physical parameters
        sat_idxs : (B,) integer satellite indices

        Returns
        -------
        (B, 3) acceleration correction in km/s^2
        """
        # Normalize state per-satellite
        state_norm = torch.empty_like(state_km)
        state_norm[:, :3] = state_km[:, :3] / r_refs.unsqueeze(1)
        state_norm[:, 3:] = state_km[:, 3:] / v_refs.unsqueeze(1)

        # a_ref = v_ref^2 / r_ref per satellite
        a_refs = v_refs ** 2 / r_refs  # (B,)

        a_norm = self.correction(state_norm, phys_params, sat_idxs)  # (B, 3)
        return a_norm * a_refs.unsqueeze(1)

    def dynamics(self, state_km: torch.Tensor,
                 r_refs: torch.Tensor, v_refs: torch.Tensor,
                 phys_params: torch.Tensor,
                 sat_idxs: torch.Tensor) -> torch.Tensor:
        """Full RHS: d/dt [pos, vel] = [vel, a_gravity + a_nn].

        Parameters
        ----------
        state_km : (B, 6) [pos_km, vel_kms]
        r_refs, v_refs : (B,) normalization references
        phys_params : (B, 4)
        sat_idxs : (B,)

        Returns
        -------
        (B, 6) [vel_kms, accel_kms2]
        """
        pos = state_km[:, :3]
        vel = state_km[:, 3:]
        a_grav = gravity_j2j5_torch(pos)
        a_nn = self.nn_acceleration(state_km, r_refs, v_refs,
                                    phys_params, sat_idxs)
        return torch.cat([vel, a_grav + a_nn], dim=1)

    def rk4_step(self, state: torch.Tensor, dt: float,
                 r_refs: torch.Tensor, v_refs: torch.Tensor,
                 phys_params: torch.Tensor,
                 sat_idxs: torch.Tensor) -> torch.Tensor:
        """Single RK4 step with conditioning passed through."""
        args = (r_refs, v_refs, phys_params, sat_idxs)
        k1 = self.dynamics(state, *args)
        k2 = self.dynamics(state + 0.5 * dt * k1, *args)
        k3 = self.dynamics(state + 0.5 * dt * k2, *args)
        k4 = self.dynamics(state + dt * k3, *args)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_batched(
        self,
        states0: torch.Tensor,
        rel_times: torch.Tensor,
        dt: float,
        r_refs: torch.Tensor,
        v_refs: torch.Tensor,
        phys_params: torch.Tensor,
        sat_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate a batch of initial states with per-satellite conditioning.

        Parameters
        ----------
        states0 : (B, 6) batch of initial states [km, km/s]
        rel_times : (M,) evaluation times relative to t=0 (seconds)
        dt : float, RK4 step size in seconds
        r_refs : (B,) reference radii
        v_refs : (B,) reference velocities
        phys_params : (B, 4) physical parameters
        sat_idxs : (B,) satellite indices

        Returns
        -------
        (B, M, 6) predicted states at each eval time
        """
        eval_np = rel_times.detach().cpu().numpy()
        results = []
        current_states = states0
        current_t = 0.0
        args = (r_refs, v_refs, phys_params, sat_idxs)

        for target_t in eval_np:
            while current_t + dt < target_t:
                current_states = self.rk4_step(current_states, dt, *args)
                current_t += dt
            remaining = target_t - current_t
            if remaining > 1e-6:
                current_states = self.rk4_step(current_states, remaining, *args)
                current_t = target_t
            results.append(current_states)

        return torch.stack(results, dim=1)  # (B, M, 6)

    def integrate_batched_hermite(
        self,
        states0: torch.Tensor,
        rel_times: torch.Tensor,
        dt: float,
        r_refs: torch.Tensor,
        v_refs: torch.Tensor,
        phys_params: torch.Tensor,
        sat_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Hermite-interpolated batch integration for fast inference.

        Parameters
        ----------
        states0 : (B, 6) batch of initial states [km, km/s]
        rel_times : (M,) evaluation times relative to t=0 (seconds)
        dt : float, RK4 step size in seconds
        r_refs : (B,) reference radii
        v_refs : (B,) reference velocities
        phys_params : (B, 4) physical parameters
        sat_idxs : (B,) satellite indices

        Returns
        -------
        (B, M, 6) predicted states at each eval time
        """
        t_end = rel_times[-1].item()
        n_steps = int(math.ceil(t_end / dt))
        args = (r_refs, v_refs, phys_params, sat_idxs)

        all_states = [states0]
        all_derivs = [self.dynamics(states0, *args)]
        current = states0
        for _ in range(n_steps):
            current = self.rk4_step(current, dt, *args)
            all_states.append(current)
            all_derivs.append(self.dynamics(current, *args))
        all_states = torch.stack(all_states, dim=0)  # (n_steps+1, B, 6)
        all_derivs = torch.stack(all_derivs, dim=0)

        step_idx = torch.clamp((rel_times / dt).long(), 0, n_steps - 1)
        tau = (rel_times - step_idx.to(rel_times.dtype) * dt) / dt
        tau = tau[:, None, None]

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
        dt: float,
        r_ref: float,
        v_ref: float,
        phys_params: torch.Tensor,
        sat_idx: int,
    ) -> torch.Tensor:
        """Integrate a single satellite from state0 at t0 to eval_times.

        Convenience wrapper: wraps scalars into batch dim=1.

        Returns
        -------
        (M, 6) states at eval_times
        """
        if state0.dim() == 1:
            state0 = state0.unsqueeze(0)
        device = state0.device
        r_refs = torch.tensor([r_ref], dtype=torch.float64, device=device)
        v_refs = torch.tensor([v_ref], dtype=torch.float64, device=device)
        pp = phys_params.unsqueeze(0) if phys_params.dim() == 1 else phys_params
        si = torch.tensor([sat_idx], dtype=torch.long, device=device)

        rel_times = eval_times - t0
        result = self.integrate_batched_hermite(
            state0, rel_times, dt, r_refs, v_refs, pp, si
        )
        return result.squeeze(0)  # (M, 6)
