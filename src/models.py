"""
Shared neural network architectures for PINN orbital propagation.

Provides:
    - FourierPINN: Fourier-featured NN with secular drift + residual connections
    - VanillaMLP:  Plain tanh MLP baseline (no Fourier features)

Architecture improvements over initial version:
    - N_FREQ: 8 -> 16  (captures RAAN precession beat frequencies)
    - HIDDEN: 64 -> 128 (avoids bottleneck with 32-feature Fourier encoding)
    - Residual connections in hidden layers (helps gradient flow through
      double-autograd physics loss computing 2nd derivatives)
"""

import torch
import torch.nn as nn


# -- Default architecture constants ------------------------------------------

N_FREQ = 16      # Fourier frequencies (doubled from 8 to capture beat freqs)
HIDDEN = 128     # Hidden layer width (doubled from 64, matches 2*N_FREQ=32 input)
LAYERS = 3       # Hidden layers


# -- Residual block -----------------------------------------------------------

class _ResidualTanhBlock(nn.Module):
    """Linear -> Tanh with additive residual connection.

    h_out = h_in + tanh(linear(h_in))

    The residual skip helps gradient flow through the double-autograd
    physics loss which computes 2nd derivatives through the full network.
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + torch.tanh(self.linear(x))


# -- FourierPINN --------------------------------------------------------------

class FourierPINN(nn.Module):
    """Fourier-featured neural network with secular drift for orbital prediction.

    Architecture:
        pos(t) = periodic_net(Fourier_enc(t))  +  sec_head(Fourier_enc(t)) * t

    The periodic backbone uses residual connections in hidden layers:
        - First layer: Linear(input_dim, hidden) -> Tanh  (no residual, dim change)
        - Hidden layers: h = h + tanh(Linear(hidden, hidden))  (residual)
        - Output layer: Linear(hidden, 3)

    The secular drift head captures slow monotonic drift of orbital elements
    (RAAN precession, argument-of-perigee precession) that manifests in ECI
    coordinates as t-modulated Fourier oscillations.

    The sec_head weights are zero-initialized so the model starts as a purely
    periodic network and only grows secular terms as training demands them.
    """

    def __init__(self, n_freq=N_FREQ, hidden=HIDDEN, n_layers=LAYERS):
        super().__init__()
        self.n_freq = n_freq
        input_dim = 2 * n_freq
        self.register_buffer(
            "freqs", torch.arange(1, n_freq + 1, dtype=torch.float64)
        )

        # Periodic backbone with residual connections
        layers = []
        # First layer: dimension change (no residual possible)
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.Tanh())
        # Hidden layers: residual connections
        for _ in range(n_layers - 1):
            layers.append(_ResidualTanhBlock(hidden))
        # Output layer
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
        return torch.cat([torch.sin(wt), torch.cos(wt)], dim=1)

    def forward(self, t):
        enc = self.encode(t)                    # (N, 2*N_FREQ)
        periodic = self.net(enc)                # (N, 3) -- purely periodic part
        secular = self.sec_head(enc) * t        # (N, 3) -- t-modulated drift part
        return periodic + secular


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
