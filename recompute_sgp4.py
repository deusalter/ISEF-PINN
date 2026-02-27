#!/usr/bin/env python3
"""Re-propagate SGP4 with updated TLEs and recompute comparison metrics.

Reads existing PINN results from data/gmat_results/*_comparison_7day.json,
re-propagates SGP4 with current TLEs in data/tle_catalog/, and updates
the SGP4 RMSE and related metrics. PINN metrics are preserved as-is
(deterministic seed=42 means they'd be identical if retrained).

Usage:
    python recompute_sgp4.py          # 5-orbit mode
    python recompute_sgp4.py --long-arc  # 7-day mode
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
from sgp4.api import Satrec

sys.path.insert(0, os.path.dirname(__file__))
from frame_conversion import teme_to_j2000_batch
from download_tle import load_tle
from satellite_catalog import get_catalog
from src.physics import MU


def propagate_sgp4_j2000(line1, line2, t_seconds, epoch_dt):
    """Propagate SGP4 and convert TEME -> J2000."""
    sat = Satrec.twoline2rv(line1, line2)
    N = len(t_seconds)

    # SGP4 uses time offsets from TLE epoch
    # t_seconds are offsets from GMAT epoch
    # Need to compute offset between TLE epoch and GMAT epoch
    jd_array = np.full(N, sat.jdsatepoch)
    fr_array = sat.jdsatepochF + t_seconds / 86400.0

    e_arr, r_arr, v_arr = sat.sgp4_array(jd_array, fr_array)

    bad = e_arr != 0
    if np.any(bad):
        n_bad = int(np.sum(bad))
        print(f"    WARNING: SGP4 had {n_bad}/{N} error points")
        for i in np.where(bad)[0]:
            if i > 0:
                r_arr[i] = r_arr[i - 1]
                v_arr[i] = v_arr[i - 1]

    pos_j2000, vel_j2000 = teme_to_j2000_batch(r_arr, v_arr, epoch_dt)
    return pos_j2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-arc", action="store_true")
    args = parser.parse_args()

    suffix = "_7day" if args.long_arc else ""
    arc_label = "7-day" if args.long_arc else "5-orbit"
    TRAIN_FRAC = 0.20

    catalog = get_catalog()

    print("=" * 70)
    print(f"  Re-propagate SGP4 with updated TLEs ({arc_label})")
    print("=" * 70)

    all_results = []

    for i, sat in enumerate(catalog, 1):
        norad_id = sat.norad_id
        name = sat.name
        orbit_type = sat.orbit_type

        print(f"\n  [{i}/{len(catalog)}] {name} (NORAD {norad_id})")

        # Load existing comparison result (for PINN metrics)
        result_path = f"data/gmat_results/{norad_id}_comparison{suffix}.json"
        if not os.path.exists(result_path):
            print(f"    SKIPPED (no existing result: {result_path})")
            continue

        with open(result_path) as f:
            result = json.load(f)

        # Load GMAT ground truth
        data_path = f"data/gmat_orbits/{norad_id}{suffix}.npy"
        meta_path = f"data/gmat_orbits/{norad_id}_meta.json"
        if not os.path.exists(data_path):
            print(f"    SKIPPED (no GMAT data)")
            continue

        gmat_data = np.load(data_path)
        with open(meta_path) as f:
            meta = json.load(f)

        t_seconds = gmat_data[:, 0]
        gmat_pos_km = gmat_data[:, 1:4]
        n_train = int(gmat_data.shape[0] * TRAIN_FRAC)

        # Load new TLE
        tle_result = load_tle(norad_id)
        if tle_result is None:
            print(f"    SKIPPED (no TLE)")
            continue
        line1, line2 = tle_result

        # Parse TLE epoch
        epoch_str = line1[18:32].strip()
        year_2d = int(epoch_str[:2])
        day_frac = float(epoch_str[2:])
        year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
        epoch_dt = datetime(year, 1, 1) + timedelta(days=day_frac - 1.0)

        # GMAT epoch
        gmat_epoch_str = meta.get("epoch_utc", "2024-01-01T12:00:00")
        gmat_epoch_dt = datetime.fromisoformat(gmat_epoch_str)

        # Compute time offset: TLE epoch vs GMAT epoch
        tle_offset_s = (gmat_epoch_dt - epoch_dt).total_seconds()
        print(f"    TLE epoch: {epoch_dt.isoformat()}")
        print(f"    GMAT epoch: {gmat_epoch_dt.isoformat()}")
        print(f"    TLE age at GMAT start: {abs(tle_offset_s)/3600:.1f}h")

        # Propagate SGP4 — t_seconds are relative to GMAT epoch,
        # but SGP4 needs time relative to TLE epoch
        t_from_tle = t_seconds + tle_offset_s
        sgp4_pos_km = propagate_sgp4_j2000(line1, line2, t_from_tle, epoch_dt)

        # Compute SGP4 errors
        sgp4_err = np.linalg.norm(sgp4_pos_km - gmat_pos_km, axis=1)
        sgp4_test_rmse = float(np.sqrt(np.mean(sgp4_err[n_train:] ** 2)))
        sgp4_train_rmse = float(np.sqrt(np.mean(sgp4_err[:n_train] ** 2)))
        sgp4_full_rmse = float(np.sqrt(np.mean(sgp4_err ** 2)))
        sgp4_max_err = float(np.max(sgp4_err[n_train:]))

        # SGP4 inference timing
        sat_sgp4 = Satrec.twoline2rv(line1, line2)
        N_pts = len(t_seconds)
        jd_array = np.full(N_pts, sat_sgp4.jdsatepoch)
        fr_array = sat_sgp4.jdsatepochF + t_from_tle / 86400.0
        _ = sat_sgp4.sgp4_array(jd_array, fr_array)  # warm-up
        sgp4_times = []
        for _ in range(10):
            sat_sgp4 = Satrec.twoline2rv(line1, line2)
            jd_array = np.full(N_pts, sat_sgp4.jdsatepoch)
            fr_array = sat_sgp4.jdsatepochF + t_from_tle / 86400.0
            t0 = time.perf_counter()
            _ = sat_sgp4.sgp4_array(jd_array, fr_array)
            sgp4_times.append((time.perf_counter() - t0) * 1e3)
        sgp4_inference_ms = float(np.median(sgp4_times))

        # Energy drift for SGP4
        dt_data = float(t_seconds[1] - t_seconds[0])
        vel = np.diff(sgp4_pos_km, axis=0) / dt_data
        v2 = np.sum(vel ** 2, axis=1)
        r = np.linalg.norm(sgp4_pos_km[:-1], axis=1)
        energy = 0.5 * v2 - MU / r
        sgp4_energy_drift = float(np.max(np.abs(energy - energy[0])))

        # Update result with new SGP4 metrics (keep PINN metrics unchanged)
        pinn_test_rmse = result["pinn_test_rmse_km"]
        improvement = (sgp4_test_rmse - pinn_test_rmse) / sgp4_test_rmse * 100 \
            if sgp4_test_rmse > 0 else 0.0

        result["sgp4_test_rmse_km"] = round(sgp4_test_rmse, 4)
        result["sgp4_train_rmse_km"] = round(sgp4_train_rmse, 4)
        result["sgp4_full_rmse_km"] = round(sgp4_full_rmse, 4)
        result["sgp4_max_err_km"] = round(sgp4_max_err, 4)
        result["sgp4_inference_ms"] = round(sgp4_inference_ms, 2)
        result["sgp4_energy_drift"] = sgp4_energy_drift
        result["pinn_improvement_over_sgp4_pct"] = round(improvement, 2)

        # Save updated result
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"    PINN test RMSE: {pinn_test_rmse:.2f} km (unchanged)")
        print(f"    SGP4 test RMSE: {sgp4_test_rmse:.2f} km (updated)")
        print(f"    SGP4 max error: {sgp4_max_err:.2f} km")
        print(f"    Improvement: {improvement:+.1f}%")

        all_results.append(result)

    # Summary
    if all_results:
        pinn_rmses = [r["pinn_test_rmse_km"] for r in all_results]
        sgp4_rmses = [r["sgp4_test_rmse_km"] for r in all_results]
        pinn_wins = sum(1 for p, s in zip(pinn_rmses, sgp4_rmses) if p < s)

        print("\n" + "=" * 70)
        print(f"  SUMMARY ({len(all_results)} satellites)")
        print("=" * 70)
        print(f"  PINN mean RMSE: {np.mean(pinn_rmses):.2f} km")
        print(f"  SGP4 mean RMSE: {np.mean(sgp4_rmses):.2f} km")
        print(f"  PINN wins: {pinn_wins}/{len(all_results)}")
        print(f"  PINN median RMSE: {np.median(pinn_rmses):.2f} km")
        print(f"  SGP4 median RMSE: {np.median(sgp4_rmses):.2f} km")

        # Save updated summary
        from scipy import stats
        pinn_arr = np.array(pinn_rmses)
        sgp4_arr = np.array(sgp4_rmses)
        diff = pinn_arr - sgp4_arr
        t_stat, p_val_two = stats.ttest_rel(pinn_arr, sgp4_arr)
        p_val_one = p_val_two / 2 if t_stat < 0 else 1.0 - p_val_two / 2
        cohens_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff) > 0 else 0.0

        summary = {
            "n_satellites": len(all_results),
            "pinn_mean_rmse_km": round(float(np.mean(pinn_rmses)), 4),
            "sgp4_mean_rmse_km": round(float(np.mean(sgp4_rmses)), 4),
            "mean_difference_km": round(float(np.mean(diff)), 4),
            "pinn_mean_max_err_km": round(float(np.mean([r["pinn_max_err_km"] for r in all_results])), 4),
            "sgp4_mean_max_err_km": round(float(np.mean([r["sgp4_max_err_km"] for r in all_results])), 4),
            "pinn_mean_inference_ms": round(float(np.mean([r["pinn_inference_ms"] for r in all_results])), 2),
            "sgp4_mean_inference_ms": round(float(np.mean([r["sgp4_inference_ms"] for r in all_results])), 2),
            "pinn_mean_energy_drift": float(np.mean([r["pinn_energy_drift"] for r in all_results])),
            "sgp4_mean_energy_drift": float(np.mean([r["sgp4_energy_drift"] for r in all_results])),
            "gmat_mean_energy_drift": float(np.mean([r["gmat_energy_drift"] for r in all_results])),
            "t_statistic": round(float(t_stat), 4),
            "p_value_one_sided": float(p_val_one),
            "cohens_d": round(cohens_d, 4),
            "significant_at_005": bool(p_val_one < 0.05),
            "pinn_wins": pinn_wins,
            "results": all_results,
        }

        summary_path = "data/gmat_results/hypothesis_test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
