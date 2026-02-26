"""
Shared neural network architectures for PINN orbital propagation.

Provides:
    - FourierPINN: Fourier-featured NN with secular drift head
    - VanillaMLP:  Plain tanh MLP baseline (no Fourier features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """

    def __init__(self, n_freq=N_FREQ, hidden=HIDDEN, n_layers=LAYERS,
                 dropout_p=0.0):
        super().__init__()
        self.n_freq = n_freq
        self.dropout_p = dropout_p
        self._mc_dropout = False
        input_dim = 2 * n_freq
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
        return torch.cat([torch.sin(wt), torch.cos(wt)], dim=1)

    def forward(self, t):
        enc = self.encode(t)                    # (N, 2*N_FREQ)

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
