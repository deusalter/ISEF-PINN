"""
generate_real_data.py -- Propagate real TLEs into dense trajectory data using SGP4.

Uses the satellite catalog (satellite_catalog.py) and TLE downloader
(download_tle.py) to produce high-cadence ephemerides for ~20 real LEO
satellites.  Each satellite is propagated for 5 orbital periods at 5000
evenly-spaced time steps.

SGP4 outputs TEME frame; for LEO over ~8h, TEME ~ ECI (<1 km difference).

Output format
-------------
For each satellite with NORAD ID `nnn`:

    data/real_orbits/{nnn}.npy       -- shape (5000, 7): [t, x, y, z, vx, vy, vz]
                                        t in seconds from TLE epoch, positions in km,
                                        velocities in km/s.
    data/real_orbits/{nnn}_meta.json -- orbital metadata (a, inc, ecc, period, etc.)

Units: km and km/s throughout (consistent with src/physics.py).
"""

import json
import math
import os
import sys
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Project root detection and sys.path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.physics import MU, R_EARTH
from satellite_catalog import get_catalog
from download_tle import load_tle

from sgp4.api import Satrec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "real_orbits")


# ---------------------------------------------------------------------------
# propagate_tle
# ---------------------------------------------------------------------------

def propagate_tle(line1: str, line2: str, n_periods: int = 5,
                  n_points: int = 5000) -> tuple:
    """Propagate a TLE using SGP4 and return trajectory data and metadata.

    Parameters
    ----------
    line1 : str
        First line of the TLE.
    line2 : str
        Second line of the TLE.
    n_periods : int
        Number of orbital periods to propagate (default 5).
    n_points : int
        Number of evenly-spaced output points (default 5000).

    Returns
    -------
    data : np.ndarray, shape (n_points, 7)
        Columns [t, x, y, z, vx, vy, vz].  t is seconds from TLE epoch,
        positions in km, velocities in km/s.
    meta : dict
        Orbital metadata: norad_id, name (empty string -- caller fills in),
        a_km, inc_deg, ecc, period_s, n_orbits, n_points.

    Raises
    ------
    RuntimeError
        If SGP4 returns non-zero error codes for any propagation point.
    ValueError
        If validation checks fail (NaN, unreasonable altitude or velocity).

    Notes
    -----
    SGP4 outputs positions and velocities in the TEME (True Equator Mean
    Equinox) frame.  For LEO satellites over ~8 hours of propagation, TEME
    is effectively identical to ECI (< 1 km difference).
    """
    # --- Parse TLE ---
    sat = Satrec.twoline2rv(line1, line2)

    # --- Compute orbital period from mean motion ---
    # sat.no_kozai is in rad/min; convert to rad/s for period computation.
    n_rad_s = sat.no_kozai / 60.0          # rad/s
    if n_rad_s <= 0:
        raise RuntimeError(f"Invalid mean motion: no_kozai={sat.no_kozai}")
    period_s = 2.0 * math.pi / n_rad_s    # seconds

    # --- Compute semi-major axis from period ---
    # a = (MU * (T / 2pi)^2)^(1/3)
    a_km = (MU * (period_s / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)

    # --- Orbital elements from TLE ---
    inc_deg = math.degrees(sat.inclo)      # inclination [deg]
    ecc = sat.ecco                         # eccentricity

    # --- Time array: 0 to n_periods * period_s, n_points steps ---
    t_end = n_periods * period_s
    t_seconds = np.linspace(0.0, t_end, n_points)

    # --- SGP4 batch propagation ---
    # sat.jdsatepoch and sat.jdsatepochF together give the TLE epoch as a
    # Julian date.  We offset from that epoch by adding t_seconds/86400 to
    # the fractional part.
    jd_base = sat.jdsatepoch
    fr_base = sat.jdsatepochF

    jd_array = np.full(n_points, jd_base)
    fr_array = fr_base + t_seconds / 86400.0

    e_arr, r_arr, v_arr = sat.sgp4_array(jd_array, fr_array)
    # r_arr: (n_points, 3) in km       (TEME frame)
    # v_arr: (n_points, 3) in km/s     (TEME frame)
    # e_arr: (n_points,) error codes -- 0 means success

    # --- Check for SGP4 errors ---
    bad_mask = e_arr != 0
    n_bad = int(np.sum(bad_mask))
    if n_bad > 0:
        bad_codes = np.unique(e_arr[bad_mask])
        raise RuntimeError(
            f"SGP4 returned {n_bad}/{n_points} error points "
            f"(codes: {bad_codes.tolist()})"
        )

    # --- Assemble output array ---
    data = np.zeros((n_points, 7), dtype=np.float64)
    data[:, 0] = t_seconds
    data[:, 1:4] = r_arr
    data[:, 4:7] = v_arr

    # --- Validation ---
    _validate_trajectory(data, a_km, ecc)

    # --- Metadata ---
    meta = {
        "norad_id": int(line2[2:7]),
        "name": "",               # caller fills in from catalog
        "a_km": float(a_km),
        "inc_deg": float(inc_deg),
        "ecc": float(ecc),
        "period_s": float(period_s),
        "n_orbits": int(n_periods),
        "n_points": int(n_points),
    }

    return data, meta


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_trajectory(data: np.ndarray, a_km: float, ecc: float) -> None:
    """Run sanity checks on a propagated trajectory.

    Checks
    ------
    1. No NaN values in positions or velocities.
    2. Radial distance is within 500 km of the expected semi-major axis.
    3. Velocity magnitude is in the 6--9 km/s range (reasonable for LEO).

    Raises
    ------
    ValueError
        If any check fails.
    """
    pos = data[:, 1:4]
    vel = data[:, 4:7]

    # 1. NaN check
    if np.any(np.isnan(pos)):
        raise ValueError("NaN detected in position data")
    if np.any(np.isnan(vel)):
        raise ValueError("NaN detected in velocity data")

    # 2. Radial distance check
    r = np.linalg.norm(pos, axis=1)
    r_expected = a_km                       # approximate, ignoring ecc
    max_dev = np.max(np.abs(r - r_expected))
    if max_dev > 500.0:
        raise ValueError(
            f"Radial distance deviates by {max_dev:.1f} km from expected "
            f"{r_expected:.1f} km (tolerance: 500 km)"
        )

    # 3. Velocity magnitude check (6--9 km/s for LEO)
    v = np.linalg.norm(vel, axis=1)
    v_min, v_max = v.min(), v.max()
    if v_min < 6.0 or v_max > 9.0:
        raise ValueError(
            f"Velocity magnitude out of LEO range: "
            f"[{v_min:.3f}, {v_max:.3f}] km/s (expected 6--9 km/s)"
        )


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------

def generate_all() -> None:
    """Iterate the satellite catalog, propagate each TLE, and save files.

    For each satellite:
      - Downloads / loads the TLE via ``load_tle(norad_id)``.
      - Propagates for 5 orbital periods at 5000 points.
      - Saves ``data/real_orbits/{norad_id}.npy`` and
        ``data/real_orbits/{norad_id}_meta.json``.

    Satellites that fail (missing TLE, SGP4 error, validation failure) are
    skipped with a warning; processing continues for the rest of the catalog.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    catalog = get_catalog()
    n_total = len(catalog)
    n_success = 0
    n_skipped = 0
    skipped_names = []

    print("=" * 65)
    print("  REAL-ORBIT DATA GENERATION (SGP4)")
    print("=" * 65)
    print(f"  Output directory : {_OUTPUT_DIR}")
    print(f"  Catalog size     : {n_total} satellites")
    print()

    for entry in catalog:
        norad_id = entry.norad_id
        name = entry.name
        print(f"[{norad_id}] {name} ... ", end="", flush=True)

        # --- Load TLE ---
        tle_result = load_tle(norad_id)
        if tle_result is None:
            print("SKIPPED (no TLE available)")
            n_skipped += 1
            skipped_names.append(f"{norad_id} {name}")
            continue

        line1, line2 = tle_result

        # --- Propagate ---
        try:
            data, meta = propagate_tle(line1, line2, n_periods=5,
                                       n_points=5000)
        except (RuntimeError, ValueError) as exc:
            print(f"SKIPPED ({exc})")
            n_skipped += 1
            skipped_names.append(f"{norad_id} {name}")
            continue

        # --- Fill in name from catalog ---
        meta["name"] = name

        # --- Save .npy ---
        npy_path = os.path.join(_OUTPUT_DIR, f"{norad_id}.npy")
        np.save(npy_path, data)

        # --- Save metadata JSON ---
        json_path = os.path.join(_OUTPUT_DIR, f"{norad_id}_meta.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        # --- Summary line ---
        period_min = meta["period_s"] / 60.0
        t_hours = data[-1, 0] / 3600.0
        r = np.linalg.norm(data[:, 1:4], axis=1)
        print(f"OK  (a={meta['a_km']:.1f} km, T={period_min:.1f} min, "
              f"inc={meta['inc_deg']:.1f} deg, ecc={meta['ecc']:.4f}, "
              f"span={t_hours:.1f} h, r=[{r.min():.1f}, {r.max():.1f}] km)")
        n_success += 1

    # --- Final summary ---
    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Processed   : {n_success}/{n_total} satellites")
    print(f"  Skipped     : {n_skipped}/{n_total}")
    if skipped_names:
        print(f"  Skipped IDs : {', '.join(skipped_names)}")
    print(f"  Output dir  : {_OUTPUT_DIR}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for real-orbit data generation."""
    generate_all()


if __name__ == "__main__":
    main()
