"""
evaluate_universal.py -- Evaluate universal NeuralODE vs SGP4 against GMAT truth
=================================================================================

Evaluates the trained universal model on all 20 satellites:
  - 16 training satellites
  - 4 held-out satellites (generalization test with zero embedding)

For each satellite:
  1. Propagate universal NeuralODE from the training boundary (20% mark)
  2. Propagate SGP4 from TLE for comparison
  3. Compute test RMSE, max error, energy drift, inference time
  4. Paired t-test: universal NeuralODE vs SGP4

Usage:
  python evaluate_universal.py                          # evaluate all 20 sats
  python evaluate_universal.py --model outputs/universal_model_valbest.pt
  python evaluate_universal.py --sat 25544              # single satellite
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

from src.physics import MU, circular_velocity
from src.models import UniversalNeuralODE
from satellite_catalog import (
    get_catalog, get_by_norad_id, get_train_catalog, get_holdout_catalog,
    SAT_TO_IDX, HOLDOUT_IDS, CATALOG,
)
from download_tle import load_tle
from frame_conversion import teme_to_j2000_batch
from sgp4.api import Satrec

# ---------------------------------------------------------------------------
# Device & constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_FRAC = 0.20
DT = 60.0


def make_phys_params(sat_entry, device):
    """Build physical parameter vector [a_km, inc_deg, ecc, cd_a_over_m]."""
    return torch.tensor([
        sat_entry.approx_alt_km + 6378.137,
        sat_entry.approx_inc_deg,
        sat_entry.approx_ecc,
        sat_entry.cd_a_over_m,
    ], dtype=torch.float64, device=device)


def propagate_sgp4_j2000(line1, line2, t_seconds, epoch_dt):
    """Propagate SGP4 and convert to J2000."""
    sat = Satrec.twoline2rv(line1, line2)
    N = len(t_seconds)
    jd_array = np.full(N, sat.jdsatepoch)
    fr_array = sat.jdsatepochF + t_seconds / 86400.0
    e_arr, r_arr, v_arr = sat.sgp4_array(jd_array, fr_array)
    bad = e_arr != 0
    if np.any(bad):
        for i in np.where(bad)[0]:
            if i > 0:
                r_arr[i] = r_arr[i - 1]
                v_arr[i] = v_arr[i - 1]
    pos_j2000, vel_j2000 = teme_to_j2000_batch(r_arr, v_arr, epoch_dt)
    return pos_j2000


def evaluate_satellite(model, norad_id, sat_entry, device):
    """Evaluate universal model on a single satellite.

    Returns dict with results, or None if data unavailable.
    """
    # Load data
    data_path = f"data/gmat_orbits/{norad_id}_7day.npy"
    meta_path = f"data/gmat_orbits/{norad_id}_meta.json"
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        print(f"  SKIP {norad_id}: no 7-day data")
        return None

    data = np.load(data_path)
    with open(meta_path) as f:
        meta = json.load(f)

    N = data.shape[0]
    t_seconds = data[:, 0]
    states_km = data[:, 1:7]
    gmat_pos_km = data[:, 1:4]
    a_km = meta["a_km"]
    v_ref = circular_velocity(a_km)
    r_ref = a_km

    n_train = int(N * TRAIN_FRAC)
    is_holdout = norad_id in HOLDOUT_IDS
    sat_idx = 0 if is_holdout else SAT_TO_IDX.get(norad_id, 0)
    pp = make_phys_params(sat_entry, device)

    print(f"  Data: {N} pts, span={t_seconds[-1]/86400:.1f} days, "
          f"n_train={n_train}")
    if is_holdout:
        print(f"  HELD-OUT: using zero embedding (index 0)")

    # --- Universal NeuralODE prediction ---
    model.eval()
    state0 = torch.tensor(states_km[n_train - 1], dtype=torch.float64,
                          device=device)
    t0_test = float(t_seconds[n_train - 1])
    t_test_tensor = torch.tensor(t_seconds[n_train:], dtype=torch.float64,
                                 device=device)

    with torch.no_grad():
        test_pred = model.integrate_to_times(
            state0, t0_test, t_test_tensor, DT,
            r_ref, v_ref, pp, sat_idx,
        )  # (N_test, 6)

    ode_pred_test = test_pred.cpu().numpy()

    # --- SGP4 prediction ---
    tle_result = load_tle(norad_id)
    if tle_result is None:
        print(f"  SKIP {norad_id}: no TLE")
        return None
    line1, line2 = tle_result

    epoch_str = line1[18:32].strip()
    year_2d = int(epoch_str[:2])
    day_frac = float(epoch_str[2:])
    year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
    epoch_dt = datetime(year, 1, 1) + timedelta(days=day_frac - 1.0)

    sgp4_pos_km = propagate_sgp4_j2000(line1, line2, t_seconds, epoch_dt)

    # --- Compute errors (test region) ---
    ode_err = np.linalg.norm(ode_pred_test[:, :3] - gmat_pos_km[n_train:], axis=1)
    sgp4_err = np.linalg.norm(sgp4_pos_km[n_train:] - gmat_pos_km[n_train:], axis=1)

    ode_test_rmse = float(np.sqrt(np.mean(ode_err ** 2)))
    sgp4_test_rmse = float(np.sqrt(np.mean(sgp4_err ** 2)))
    ode_max_err = float(np.max(ode_err))
    sgp4_max_err = float(np.max(sgp4_err))

    # Energy drift
    dt_data = float(t_seconds[1] - t_seconds[0])

    def _energy_drift(pos_arr):
        vel = np.diff(pos_arr, axis=0) / dt_data
        v2 = np.sum(vel ** 2, axis=1)
        r = np.linalg.norm(pos_arr[:-1], axis=1)
        energy = 0.5 * v2 - MU / r
        return float(np.max(np.abs(energy - energy[0])))

    ode_energy = _energy_drift(ode_pred_test[:, :3])

    # Inference time (universal NeuralODE)
    n_timing = 3
    with torch.no_grad():
        _ = model.integrate_to_times(state0, t0_test, t_test_tensor[:500],
                                     DT, r_ref, v_ref, pp, sat_idx)
    ode_times = []
    t_sub = t_test_tensor[:min(500, len(t_test_tensor))]
    for _ in range(n_timing):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model.integrate_to_times(state0, t0_test, t_sub,
                                         DT, r_ref, v_ref, pp, sat_idx)
        ode_times.append((time.perf_counter() - t_start) * 1e3)
    scale = len(t_test_tensor) / len(t_sub)
    ode_ms = float(np.median(ode_times) * scale)

    # SGP4 inference time
    sat_sgp4 = Satrec.twoline2rv(line1, line2)
    N_pts = len(t_seconds)
    jd_array = np.full(N_pts, sat_sgp4.jdsatepoch)
    fr_array = sat_sgp4.jdsatepochF + t_seconds / 86400.0
    _ = sat_sgp4.sgp4_array(jd_array, fr_array)
    sgp4_times = []
    for _ in range(10):
        sat_sgp4 = Satrec.twoline2rv(line1, line2)
        jd_array = np.full(N_pts, sat_sgp4.jdsatepoch)
        fr_array = sat_sgp4.jdsatepochF + t_seconds / 86400.0
        t_start = time.perf_counter()
        _ = sat_sgp4.sgp4_array(jd_array, fr_array)
        sgp4_times.append((time.perf_counter() - t_start) * 1e3)
    sgp4_ms = float(np.median(sgp4_times))

    improvement = ((sgp4_test_rmse - ode_test_rmse) / sgp4_test_rmse * 100
                   if sgp4_test_rmse > 0 else 0.0)

    print(f"  Universal NODE test RMSE: {ode_test_rmse:.2f} km")
    print(f"  SGP4 test RMSE:           {sgp4_test_rmse:.2f} km")
    print(f"  Improvement:              {improvement:.1f}%")
    print(f"  NODE max error:           {ode_max_err:.2f} km")
    print(f"  NODE inference:           {ode_ms:.1f} ms")

    return {
        "norad_id": norad_id,
        "name": sat_entry.name,
        "orbit_type": sat_entry.orbit_type,
        "is_holdout": is_holdout,
        "a_km": a_km,
        "universal_test_rmse_km": round(ode_test_rmse, 4),
        "sgp4_test_rmse_km": round(sgp4_test_rmse, 4),
        "universal_max_err_km": round(ode_max_err, 4),
        "sgp4_max_err_km": round(sgp4_max_err, 4),
        "universal_energy_drift": ode_energy,
        "universal_inference_ms": round(ode_ms, 2),
        "sgp4_inference_ms": round(sgp4_ms, 2),
        "improvement_pct": round(improvement, 2),
    }


def generate_figures(train_results, holdout_results):
    """Generate comparison figures for the universal model."""
    os.makedirs("figures", exist_ok=True)
    all_results = train_results + holdout_results

    if not all_results:
        return

    # --- Figure 1: Bar chart (all satellites, color-coded by holdout) ---
    fig, ax = plt.subplots(figsize=(16, 7))
    names = [r["name"][:15] for r in all_results]
    ode_rmses = [r["universal_test_rmse_km"] for r in all_results]
    sgp4_rmses = [r["sgp4_test_rmse_km"] for r in all_results]
    is_holdout = [r["is_holdout"] for r in all_results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, sgp4_rmses, width, label="SGP4",
           color="#d62728", alpha=0.85)

    # Color universal bars: blue for train, green for holdout
    colors = ["#2ca02c" if h else "#1f77b4" for h in is_holdout]
    bars = ax.bar(x + width/2, ode_rmses, width, color=colors, alpha=0.85)

    # Legend entries
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.85, label="SGP4"),
        Patch(facecolor="#1f77b4", alpha=0.85, label="Universal NODE (train)"),
        Patch(facecolor="#2ca02c", alpha=0.85, label="Universal NODE (held-out)"),
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    ax.set_xlabel("Satellite", fontsize=12)
    ax.set_ylabel("Test RMSE (km)", fontsize=12)
    ax.set_title("Universal NeuralODE vs SGP4 (GMAT Ground Truth, 7-Day)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/universal_vs_sgp4_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/universal_vs_sgp4_comparison.png")

    # --- Figure 2: Scatter plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    max_val = max(max(ode_rmses), max(sgp4_rmses)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal accuracy")

    train_x = [r["sgp4_test_rmse_km"] for r in train_results]
    train_y = [r["universal_test_rmse_km"] for r in train_results]
    hold_x = [r["sgp4_test_rmse_km"] for r in holdout_results]
    hold_y = [r["universal_test_rmse_km"] for r in holdout_results]

    ax.scatter(train_x, train_y, s=80, c="#1f77b4", edgecolors="black",
               linewidth=0.5, zorder=5, label="Train satellites")
    ax.scatter(hold_x, hold_y, s=120, c="#2ca02c", edgecolors="black",
               linewidth=1.0, zorder=6, marker="D", label="Held-out satellites")

    for r in all_results:
        ax.annotate(r["name"][:12],
                    (r["sgp4_test_rmse_km"], r["universal_test_rmse_km"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("SGP4 Test RMSE (km)", fontsize=12)
    ax.set_ylabel("Universal NODE Test RMSE (km)", fontsize=12)
    ax.set_title("Universal NeuralODE vs SGP4: Per-Satellite", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/universal_vs_sgp4_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figures/universal_vs_sgp4_scatter.png")

    # --- Figure 3: Train vs Holdout box plot ---
    if holdout_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        train_imp = [r["improvement_pct"] for r in train_results]
        hold_imp = [r["improvement_pct"] for r in holdout_results]
        bp = ax.boxplot([train_imp, hold_imp],
                        labels=["Train sats", "Held-out sats"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#1f77b4")
        bp["boxes"][1].set_facecolor("#2ca02c")
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Improvement over SGP4 (%)", fontsize=12)
        ax.set_title("Generalization: Train vs Held-Out Satellites", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig("figures/universal_generalization_boxplot.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: figures/universal_generalization_boxplot.png")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate universal NeuralODE vs SGP4"
    )
    parser.add_argument("--model", type=str,
                        default="outputs/universal_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--sat", type=int, default=None,
                        help="Evaluate single satellite by NORAD ID")
    args = parser.parse_args()

    print("=" * 70)
    print("  Universal NeuralODE Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model:  {args.model}")

    if not os.path.exists(args.model):
        print(f"\nERROR: Model not found at {args.model}")
        print("Run train_universal.py first.")
        sys.exit(1)

    # Load model
    n_sats = len(CATALOG) + 1
    model = UniversalNeuralODE(
        n_satellites=n_sats, embed_dim=4,
        hidden=64, n_layers=2,
    ).double().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params} parameters")

    # Build satellite list
    if args.sat is not None:
        entry = get_by_norad_id(args.sat)
        if entry is None:
            print(f"ERROR: NORAD ID {args.sat} not in catalog")
            sys.exit(1)
        satellites = [entry]
    else:
        satellites = get_catalog()

    print(f"Evaluating {len(satellites)} satellites\n")

    train_results = []
    holdout_results = []

    for idx, sat in enumerate(satellites):
        print(f"\n{'='*60}")
        tag = " [HELD-OUT]" if sat.norad_id in HOLDOUT_IDS else ""
        print(f"  [{idx+1}/{len(satellites)}] {sat.name} "
              f"(NORAD {sat.norad_id}){tag}")
        print(f"{'='*60}")

        result = evaluate_satellite(model, sat.norad_id, sat, DEVICE)
        if result is not None:
            if result["is_holdout"]:
                holdout_results.append(result)
            else:
                train_results.append(result)

    # --- Save all results ---
    all_results = train_results + holdout_results
    os.makedirs("data/gmat_results", exist_ok=True)
    with open("data/gmat_results/universal_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Aggregate statistics ---
    if len(all_results) < 2:
        print("\nNot enough results for statistical analysis.")
        return

    print(f"\n\n{'='*70}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*70}")

    def _print_group(label, results):
        if not results:
            return
        ode_rmses = np.array([r["universal_test_rmse_km"] for r in results])
        sgp4_rmses = np.array([r["sgp4_test_rmse_km"] for r in results])
        imps = np.array([r["improvement_pct"] for r in results])
        print(f"\n  {label} (N={len(results)}):")
        print(f"    Universal NODE RMSE:  mean={np.mean(ode_rmses):.2f}, "
              f"median={np.median(ode_rmses):.2f} km")
        print(f"    SGP4 RMSE:            mean={np.mean(sgp4_rmses):.2f}, "
              f"median={np.median(sgp4_rmses):.2f} km")
        print(f"    Improvement over SGP4: mean={np.mean(imps):.1f}%, "
              f"median={np.median(imps):.1f}%")

        if len(results) >= 3:
            t_stat, p_two = stats.ttest_rel(ode_rmses, sgp4_rmses)
            p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
            print(f"    Paired t-test: t={t_stat:.3f}, p(one-sided)={p_one:.2e}")
            if p_one < 0.05:
                print(f"    -> SIGNIFICANT at alpha=0.05")
            else:
                print(f"    -> NOT significant at alpha=0.05")

    _print_group("TRAINING SATELLITES", train_results)
    _print_group("HELD-OUT SATELLITES (generalization)", holdout_results)
    _print_group("ALL SATELLITES", all_results)

    # Per-satellite summary table
    print(f"\n\n  {'Satellite':<25} {'Type':<18} {'Hold?':<6} "
          f"{'NODE RMSE':>10} {'SGP4 RMSE':>10} {'Improv%':>8}")
    print(f"  {'-'*25} {'-'*18} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for r in all_results:
        h = "YES" if r["is_holdout"] else ""
        print(f"  {r['name']:<25} {r['orbit_type']:<18} {h:<6} "
              f"{r['universal_test_rmse_km']:>10.2f} "
              f"{r['sgp4_test_rmse_km']:>10.2f} "
              f"{r['improvement_pct']:>7.1f}%")

    # --- Generate figures ---
    print(f"\nGenerating figures...")
    generate_figures(train_results, holdout_results)

    print(f"\nResults saved to data/gmat_results/universal_evaluation.json")
    print("Done.")


if __name__ == "__main__":
    main()
