"""PINN with Fourier feature encoding for periodic orbital dynamics.

Key insight: circular orbits are periodic (x=cos(wt), y=sin(wt)). Standard MLPs
with tanh can't extrapolate periodic functions. Fourier features give the network
natural building blocks for periodicity, dramatically improving extrapolation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

from src.physics import NormalizationParams

# Config
HIDDEN    = 64
LAYERS    = 3
N_FREQ    = 8       # Number of Fourier frequencies
WARMUP    = 1000    # Fewer warmup needed with better features
TOTAL     = 5000    # Fewer total epochs needed
LR        = 1e-3
LAMBDA    = 0.1
N_COL     = 200
CHUNK     = 1000    # Epochs per training chunk (for checkpointing)

norm = NormalizationParams()


class FourierPINN(nn.Module):
    """PINN with Fourier feature input encoding.

    Instead of raw time t, input is:
    [sin(t), cos(t), sin(2t), cos(2t), ..., sin(Nt), cos(Nt), t]
    = 2*N_FREQ + 1 input features
    """
    def __init__(self, n_freq=N_FREQ, hidden=HIDDEN, n_layers=LAYERS):
        super().__init__()
        input_dim = 2 * n_freq + 1  # sin + cos for each freq, plus raw t
        self.n_freq = n_freq
        # Frequencies: 1, 2, ..., n_freq (in normalized units, freq=1 ≈ orbital freq)
        self.register_buffer('freqs',
            torch.arange(1, n_freq + 1, dtype=torch.float64))

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, t):
        """Fourier feature encoding: t → [sin(w1*t), cos(w1*t), ..., t]"""
        # t shape: (N, 1), freqs shape: (n_freq,)
        wt = t * self.freqs  # (N, n_freq) via broadcasting
        features = torch.cat([torch.sin(wt), torch.cos(wt), t], dim=1)
        return features  # (N, 2*n_freq + 1)

    def forward(self, t):
        return self.net(self.encode(t))


def derivs(model, t):
    """Compute pos and acceleration via double autograd."""
    pos = model(t)
    vel, acc = [], []
    for i in range(3):
        v = torch.autograd.grad(pos[:,i], t, torch.ones_like(pos[:,i]),
                                create_graph=True, retain_graph=True)[0]
        vel.append(v)
    vel = torch.cat(vel, dim=1)
    for i in range(3):
        a = torch.autograd.grad(vel[:,i], t, torch.ones_like(vel[:,i]),
                                create_graph=True, retain_graph=True)[0]
        acc.append(a)
    return pos, torch.cat(acc, dim=1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=TOTAL)
    parser.add_argument('--warmup', type=int, default=WARMUP)
    args = parser.parse_args()

    data = np.load('data/orbital_data.npy')
    N = data.shape[0]
    t_raw, pos_km = data[:, 0], data[:, 1:4]
    t_norm = norm.normalize_time(t_raw)
    pos_norm = pos_km / norm.r_ref
    n_train = int(N * 0.8)

    t_tr  = torch.tensor(t_norm[:n_train, None], dtype=torch.float64)
    pos_tr = torch.tensor(pos_norm[:n_train], dtype=torch.float64)
    t_all  = torch.tensor(t_norm[:, None], dtype=torch.float64)

    t0 = t_norm[0] + 1e-4 * (t_norm[-1] - t_norm[0])
    t_col = torch.linspace(float(t0), float(t_norm[-1]), N_COL,
                            dtype=torch.float64).unsqueeze(1).requires_grad_(True)

    model = FourierPINN().double()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FourierPINN: {n_params} params, {N_FREQ} frequencies, input_dim={2*N_FREQ+1}", flush=True)
    print(f"Train: {n_train}, Test: {N-n_train}, Col: {N_COL}", flush=True)

    # Check for checkpoint
    ckpt_path = 'models/pinn_fourier_ckpt.pt'
    start_epoch = 1
    hist = []
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        hist = list(ckpt.get('history', []))
        print(f"Resumed from epoch {start_epoch - 1}", flush=True)

    mse = nn.MSELoss()
    model.train()

    for ep in range(start_epoch, args.epochs + 1):
        opt.zero_grad()
        pred = model(t_tr)
        dl = mse(pred, pos_tr)

        if ep > args.warmup:
            pc, ac = derivs(model, t_col)
            r = torch.norm(pc, dim=1, keepdim=True).clamp(min=1e-3)
            pl = torch.mean((ac + pc / (r ** 3)) ** 2)
            total = dl + LAMBDA * pl
            pv = pl.item()
        else:
            total = dl
            pv = 0.0

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append([total.item(), dl.item(), pv])

        if ep % 500 == 0 or ep == start_epoch:
            ph = "WARM" if ep <= args.warmup else "PINN"
            print(f"  [{ph}] {ep:5d}/{args.epochs}  d={dl.item():.4e}  p={pv:.4e}", flush=True)

        # Checkpoint every CHUNK epochs
        if ep % CHUNK == 0:
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'epoch': ep, 'history': hist}, ckpt_path)

    # Final save
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'epoch': args.epochs, 'history': hist}, ckpt_path)

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_n = model(t_all).numpy()
    pred_km = pred_n * norm.r_ref
    err = np.linalg.norm(pred_km - pos_km, axis=1)
    rt = np.sqrt(np.mean(err[:n_train]**2))
    re = np.sqrt(np.mean(err[n_train:]**2))
    print(f"\n  Train RMSE: {rt:.1f} km | Test RMSE: {re:.1f} km", flush=True)
    print(f"  Vanilla baseline: 10199 km", flush=True)
    print(f"  Improvement: {(1-re/10199)*100:.1f}%", flush=True)

    # Save outputs
    torch.save(model.state_dict(), 'models/pinn_twobody.pt')
    np.save('data/pinn_predictions.npy', np.column_stack([t_raw, pred_km]))
    np.save('data/pinn_loss_history.npy', np.array(hist))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pos_km[:,0], pos_km[:,1], 'b-', lw=1.5, label='Ground Truth')
    van = np.load('data/vanilla_predictions.npy')
    ax.plot(van[:,1], van[:,2], 'r:', lw=1, label='Vanilla NN')
    ax.plot(pred_km[:,0], pred_km[:,1], 'g--', lw=1.2, label=f'PINN (test={re:.0f}km)')
    ax.axvline(pos_km[n_train,0], color='k', ls='--', alpha=0.3)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
    ax.legend(); plt.tight_layout()
    plt.savefig('figures/pinn_vs_vanilla_xy.png', dpi=150)
    print("All saved!", flush=True)


if __name__ == '__main__':
    main()
