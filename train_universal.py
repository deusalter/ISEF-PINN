"""
train_universal.py -- Train one universal NeuralODE for all LEO satellites
=========================================================================

Trains a single UniversalNeuralODE on 16 training satellites simultaneously.
Uses multiple-shooting with mixed-satellite mini-batches.

Usage:
  python train_universal.py                    # full training (16 sats, 3000 epochs)
  python train_universal.py --smoke            # smoke test (3 sats, 200 epochs)
  python train_universal.py --epochs 500       # custom epoch count
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from src.physics import circular_velocity, R_EARTH
from src.models import UniversalNeuralODE
from satellite_catalog import get_train_catalog, SAT_TO_IDX, CATALOG

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.20          # 20% train, 80% test
N_SEGMENTS_PER_SAT = 20    # multiple-shooting segments per satellite
DT = 60.0                  # RK4 step size (seconds)
VEL_WEIGHT = 0.1           # velocity loss weight
GRAD_CLIP = 1.0
HIDDEN = 64
N_LAYERS = 2
EMBED_DIM = 4
EMBED_DROPOUT = 0.2
LR = 3e-4
LR_MIN = 1e-6
EPOCHS = 3000
BATCH_SIZE = 80            # segments per mini-batch
VAL_INTERVAL = 100         # validate every N epochs

# Validation satellites: 2 held-in satellites for monitoring during training
VAL_NORAD_IDS = [25544, 44713]  # ISS, STARLINK-1007


def load_satellite_data(norad_id):
    """Load 7-day GMAT data for a satellite.

    Returns
    -------
    data : np.ndarray (N, 7) [t_seconds, x, y, z, vx, vy, vz]
    meta : dict with 'a_km' key
    or (None, None) if not found
    """
    data_path = f"data/gmat_orbits/{norad_id}_7day.npy"
    meta_path = f"data/gmat_orbits/{norad_id}_7day_meta.json"
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        return None, None
    data = np.load(data_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return data, meta


def make_phys_params(sat_entry):
    """Build physical parameter vector [a_km, inc_deg, ecc, cd_a_over_m]."""
    return torch.tensor([
        sat_entry.approx_alt_km + R_EARTH,
        sat_entry.approx_inc_deg,
        sat_entry.approx_ecc,
        sat_entry.cd_a_over_m,
    ], dtype=torch.float64)


def prepare_segments(catalog, device):
    """Load data and build training segments for all satellites.

    Returns
    -------
    segments : list of dicts with keys:
        s0 (6,), rel_times (P,), truth (P, 6), r_ref, v_ref,
        phys_params (4,), sat_idx, norad_id
    val_data : dict mapping norad_id -> (full_data_tensor, n_train, r_ref, v_ref,
               phys_params, sat_idx)
    """
    segments = []
    val_data = {}

    for sat in catalog:
        data, meta = load_satellite_data(sat.norad_id)
        if data is None:
            print(f"  SKIP {sat.norad_id} ({sat.name}): no 7-day data")
            continue

        N = data.shape[0]
        t_seconds = data[:, 0]
        states_km = data[:, 1:7]

        a_km = meta["a_km"]
        v_ref = circular_velocity(a_km)
        r_ref = a_km
        sat_idx = SAT_TO_IDX.get(sat.norad_id, 0)
        pp = make_phys_params(sat)

        n_train = int(N * TRAIN_FRAC)
        train_t = t_seconds[:n_train]
        train_states = states_km[:n_train]

        # Build segments
        pts_per_seg = max(1, n_train // N_SEGMENTS_PER_SAT)
        seg_starts = list(range(0, n_train, pts_per_seg))
        if not seg_starts:
            seg_starts = [0]

        for si in seg_starts:
            se = min(si + pts_per_seg, n_train)
            s0 = torch.tensor(train_states[si], dtype=torch.float64, device=device)
            t_seg = torch.tensor(train_t[si:se], dtype=torch.float64, device=device)
            rel_t = t_seg - t_seg[0]
            truth = torch.tensor(train_states[si:se], dtype=torch.float64, device=device)
            segments.append({
                "s0": s0,
                "rel_times": rel_t,
                "truth": truth,
                "r_ref": r_ref,
                "v_ref": v_ref,
                "phys_params": pp.to(device),
                "sat_idx": sat_idx,
                "norad_id": sat.norad_id,
            })

        # Save val data if this is a validation satellite
        if sat.norad_id in VAL_NORAD_IDS:
            val_data[sat.norad_id] = {
                "states": torch.tensor(states_km, dtype=torch.float64, device=device),
                "t_seconds": torch.tensor(t_seconds, dtype=torch.float64, device=device),
                "n_train": n_train,
                "r_ref": r_ref,
                "v_ref": v_ref,
                "phys_params": pp.to(device),
                "sat_idx": sat_idx,
            }

        print(f"  {sat.norad_id:>5} {sat.name:<25} {len(seg_starts):>3} segs, "
              f"n_train={n_train}, a_km={a_km:.1f}")

    # Validate that all segments share the same time cadence
    if segments:
        cadences = set()
        for s in segments:
            rt = s["rel_times"]
            if len(rt) >= 2:
                cadences.add(round(float(rt[1] - rt[0]), 4))
        if len(cadences) > 1:
            print(f"  WARNING: non-uniform cadences across segments: {cadences}")

    return segments, val_data


def collate_batch(segments, min_pts, common_rel_times):
    """Stack a list of segment dicts into batched tensors.

    All segments are truncated to min_pts time points.
    Uses a pre-validated common relative time grid (all satellites share
    the same GMAT cadence of ~12.1s).

    Returns
    -------
    s0 : (B, 6)
    rel_times : (min_pts,)
    truth : (B, min_pts, 6)
    r_refs : (B,)
    v_refs : (B,)
    phys_params : (B, 4)
    sat_idxs : (B,)
    """
    device = segments[0]["s0"].device
    s0 = torch.stack([s["s0"] for s in segments])
    rel_times = common_rel_times[:min_pts]
    truth = torch.stack([s["truth"][:min_pts] for s in segments])
    r_refs = torch.tensor([s["r_ref"] for s in segments],
                          dtype=torch.float64, device=device)
    v_refs = torch.tensor([s["v_ref"] for s in segments],
                          dtype=torch.float64, device=device)
    phys_params = torch.stack([s["phys_params"] for s in segments])
    sat_idxs = torch.tensor([s["sat_idx"] for s in segments],
                            dtype=torch.long, device=device)
    return s0, rel_times, truth, r_refs, v_refs, phys_params, sat_idxs


def validate(model, val_data, dt):
    """Compute test-region position RMSE for validation satellites."""
    model.eval()
    results = {}
    with torch.no_grad():
        for nid, vd in val_data.items():
            n_train = vd["n_train"]
            state0 = vd["states"][n_train - 1]
            t0 = vd["t_seconds"][n_train - 1].item()
            t_test = vd["t_seconds"][n_train:]
            truth_test = vd["states"][n_train:]

            pred = model.integrate_to_times(
                state0, t0, t_test, dt,
                vd["r_ref"], vd["v_ref"],
                vd["phys_params"], vd["sat_idx"],
            )
            rmse = torch.sqrt(
                torch.mean((pred[:, :3] - truth_test[:, :3]) ** 2)
            ).item()
            results[nid] = rmse
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description="Train universal NeuralODE")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 3 sats, 200 epochs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    epochs = args.epochs or (200 if args.smoke else EPOCHS)
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  Universal NeuralODE Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {args.lr}")
    print()

    # Load catalog
    catalog = get_train_catalog()
    if args.smoke:
        catalog = catalog[:3]
        print(f"SMOKE TEST: using {len(catalog)} satellites")

    print(f"Loading data for {len(catalog)} training satellites...")
    segments, val_data = prepare_segments(catalog, DEVICE)
    print(f"\nTotal segments: {len(segments)}")
    if not segments:
        print("ERROR: No segments loaded. Check GMAT data files.")
        sys.exit(1)

    # Find minimum segment length and build common relative time grid
    min_pts = min(s["truth"].shape[0] for s in segments)
    print(f"Min pts/segment: {min_pts}")

    # All GMAT data shares ~12.1s cadence, so use first segment's grid
    common_rel_times = segments[0]["rel_times"][:min_pts]

    # Build model
    n_sats = len(CATALOG) + 1  # +1 for index 0 (unseen)
    model = UniversalNeuralODE(
        n_satellites=n_sats, embed_dim=EMBED_DIM,
        hidden=HIDDEN, n_layers=N_LAYERS,
        embed_dropout=EMBED_DROPOUT,
    ).double().to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_corr_params = sum(p.numel() for p in model.correction.net.parameters())
    n_embed_params = model.correction.embedding.weight.numel()
    print(f"Model: {n_params} total params ({n_corr_params} MLP, "
          f"{n_embed_params} embedding)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR_MIN
    )

    best_state = None
    best_loss = float("inf")
    best_val_rmse = float("inf")
    train_start = time.time()

    os.makedirs("outputs", exist_ok=True)
    log_path = "outputs/train_universal.log"
    log_file = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        log(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'='*70}")

        model.train()
        for ep in range(1, epochs + 1):
            # Shuffle segments and form mini-batches
            perm = np.random.permutation(len(segments))
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(segments), batch_size):
                batch_idx = perm[batch_start:batch_start + batch_size]
                batch_segs = [segments[i] for i in batch_idx]

                s0, rel_times, truth, r_refs, v_refs, pp, si = collate_batch(
                    batch_segs, min_pts, common_rel_times
                )

                optimizer.zero_grad()

                pred = model.integrate_batched_hermite(
                    s0, rel_times, DT, r_refs, v_refs, pp, si
                )  # (B, min_pts, 6)

                # Normalize losses by reference scales so position and velocity
                # contribute comparably (raw km vs km/s differ by ~1000x)
                pos_err = (pred[:, :, :3] - truth[:, :, :3]) / r_refs[:, None, None]
                vel_err = (pred[:, :, 3:] - truth[:, :, 3:]) / v_refs[:, None, None]
                pos_loss = torch.mean(pos_err ** 2)
                vel_loss = torch.mean(vel_err ** 2)
                total_loss = pos_loss + VEL_WEIGHT * vel_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Checkpoint best training loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Logging
            if ep % 50 == 0 or ep == 1 or ep == epochs:
                lr_now = scheduler.get_last_lr()[0]
                log_s = model.correction.log_scale.item()
                msg = (f"  ep={ep:>5}/{epochs}  loss={avg_loss:.4e}  "
                       f"best={best_loss:.4e}  lr={lr_now:.1e}  "
                       f"log_scale={log_s:.2f}")
                log(msg)

            # Validation
            if ep % VAL_INTERVAL == 0 or ep == epochs:
                if val_data:
                    val_results = validate(model, val_data, DT)
                    val_msg = "  VAL: " + ", ".join(
                        f"{nid}={rmse:.2f}km" for nid, rmse in val_results.items()
                    )
                    log(val_msg)
                    mean_val = np.mean(list(val_results.values()))
                    if mean_val < best_val_rmse:
                        best_val_rmse = mean_val
                        # Save val-best checkpoint
                        torch.save(model.state_dict(),
                                   "outputs/universal_model_valbest.pt")
                    model.train()

        elapsed = time.time() - train_start
        log(f"\nTraining completed in {elapsed/60:.1f} minutes")
        log(f"Best training loss: {best_loss:.4e}")
        if val_data:
            log(f"Best validation RMSE: {best_val_rmse:.2f} km")

        # Restore best model: prefer validation-best, fall back to train-best
        valbest_path = "outputs/universal_model_valbest.pt"
        if os.path.exists(valbest_path) and best_val_rmse < float("inf"):
            model.load_state_dict(torch.load(valbest_path, weights_only=True))
            log(f"Restored validation-best checkpoint (RMSE={best_val_rmse:.2f} km)")
        elif best_state is not None:
            model.load_state_dict(best_state)
            log("Restored training-best checkpoint (no val-best available)")

        save_path = "outputs/universal_model.pt"
        torch.save(model.state_dict(), save_path)
        log(f"Model saved to {save_path}")

        # Save training config
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "lr_min": LR_MIN,
            "hidden": HIDDEN,
            "n_layers": N_LAYERS,
            "embed_dim": EMBED_DIM,
            "embed_dropout": EMBED_DROPOUT,
            "dt": DT,
            "vel_weight": VEL_WEIGHT,
            "n_segments_per_sat": N_SEGMENTS_PER_SAT,
            "train_frac": TRAIN_FRAC,
            "n_train_sats": len(catalog),
            "n_segments": len(segments),
            "best_loss": best_loss,
            "best_val_rmse": best_val_rmse,
            "elapsed_minutes": elapsed / 60,
            "device": str(DEVICE),
            "smoke": args.smoke,
        }
        with open("outputs/universal_train_config.json", "w") as f:
            json.dump(config, f, indent=2)

    finally:
        log_file.close()

    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
