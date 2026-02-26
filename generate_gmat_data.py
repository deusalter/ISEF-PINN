"""
generate_gmat_data.py -- Generate high-fidelity trajectory data using NASA GMAT
================================================================================

Replaces SGP4 as ground truth for the PINN vs SGP4 hypothesis test.

For each satellite in the catalog:
  1. Load TLE, compute osculating Cartesian state at epoch via SGP4
  2. Convert TEME -> J2000 via frame_conversion.py
  3. Generate a GMAT .script file with a high-fidelity force model:
     - 20x20 JGM-2 gravity harmonics
     - MSISE-90 atmospheric drag
     - Solar radiation pressure (SRP)
     - Sun + Moon third-body perturbations
     - RungeKutta89 integrator, accuracy 1e-12
  4. Run GmatConsole --run --exit via subprocess
  5. Parse the ReportFile output
  6. Resample to exactly 5000 evenly-spaced points (cubic interpolation)
  7. Save as data/gmat_orbits/{norad_id}.npy (shape 5000x7)
  8. Save metadata as data/gmat_orbits/{norad_id}_meta.json

Usage:
  python generate_gmat_data.py                 # full catalog
  python generate_gmat_data.py --sat 25544     # single satellite (ISS)
  python generate_gmat_data.py --dry-run       # generate scripts only, no GMAT
  python generate_gmat_data.py --validate      # test GMAT installation
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Project root detection and sys.path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.physics import MU, R_EARTH
from satellite_catalog import get_catalog, get_by_norad_id
from download_tle import load_tle
from frame_conversion import teme_to_j2000
from gmat_config import get_gmat_console_path, get_gmat_root, validate_gmat_install

from sgp4.api import Satrec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = _PROJECT_ROOT / "data" / "gmat_orbits"
SCRIPT_DIR = _PROJECT_ROOT / "data" / "gmat_scripts"
N_POINTS = 5000          # output points (matches SGP4 pipeline)
N_ORBITS = 5             # propagation duration in orbital periods
GMAT_STEP_SEC = 10.0     # GMAT output cadence (seconds)

# Long-arc (7-day) settings
N_POINTS_LONG = 50000    # 10x more points for 10x longer arc
LONG_ARC_DAYS = 7        # propagation duration in days


# ---------------------------------------------------------------------------
# TLE epoch parsing
# ---------------------------------------------------------------------------

def _tle_epoch_to_datetime(line1: str) -> datetime:
    """Parse TLE epoch from line 1 into a datetime object.

    TLE epoch format: columns 18-32, YYDDD.DDDDDDDD
    where YY = 2-digit year, DDD.DDD = fractional day of year.
    """
    epoch_str = line1[18:32].strip()
    year_2d = int(epoch_str[:2])
    day_frac = float(epoch_str[2:])

    # 2-digit year: 00-56 -> 2000-2056, 57-99 -> 1957-1999
    year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d

    # Day 1 = Jan 1
    epoch_dt = datetime(year, 1, 1) + timedelta(days=day_frac - 1.0)
    return epoch_dt


def _datetime_to_gmat_epoch(dt: datetime) -> str:
    """Convert datetime to GMAT's UTCGregorian format.

    Format: 'DD Mon YYYY HH:MM:SS.mmm'
    Example: '01 Jan 2024 12:00:00.000'
    """
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    ms = dt.microsecond / 1000.0
    return (
        f"{dt.day:02d} {months[dt.month - 1]} {dt.year} "
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03.0f}"
    )


# ---------------------------------------------------------------------------
# SGP4 state extraction
# ---------------------------------------------------------------------------

def get_sgp4_state_at_epoch(line1: str, line2: str):
    """Get position/velocity from SGP4 at TLE epoch.

    Returns
    -------
    pos_km : np.ndarray, shape (3,)
        Position in TEME frame [km].
    vel_kms : np.ndarray, shape (3,)
        Velocity in TEME frame [km/s].
    epoch_dt : datetime
        TLE epoch as datetime.
    period_s : float
        Orbital period [seconds].
    a_km : float
        Semi-major axis [km].
    inc_deg : float
        Inclination [degrees].
    ecc : float
        Eccentricity.
    """
    sat = Satrec.twoline2rv(line1, line2)

    # Propagate at exactly the TLE epoch (t=0)
    e, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if e != 0:
        raise RuntimeError(f"SGP4 error code {e} at epoch")

    pos_km = np.array(r, dtype=np.float64)
    vel_kms = np.array(v, dtype=np.float64)
    epoch_dt = _tle_epoch_to_datetime(line1)

    # Orbital elements
    n_rad_s = sat.no_kozai / 60.0  # rad/s
    period_s = 2.0 * math.pi / n_rad_s
    a_km = (MU * (period_s / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
    inc_deg = math.degrees(sat.inclo)
    ecc = sat.ecco

    return pos_km, vel_kms, epoch_dt, period_s, a_km, inc_deg, ecc


# ---------------------------------------------------------------------------
# GMAT script generation
# ---------------------------------------------------------------------------

def generate_gmat_script(
    norad_id: int,
    name: str,
    pos_j2000: np.ndarray,
    vel_j2000: np.ndarray,
    epoch_dt: datetime,
    period_s: float,
    a_km: float,
    ecc: float,
    cd_a_over_m: float = 0.022,
    prop_duration_override: float = None,
    script_suffix: str = "",
) -> Path:
    """Generate a GMAT .script file for high-fidelity propagation.

    Parameters
    ----------
    norad_id : int
        NORAD catalog ID.
    name : str
        Satellite name (used in comments only).
    pos_j2000 : np.ndarray, shape (3,)
        Initial position in J2000 frame [km].
    vel_j2000 : np.ndarray, shape (3,)
        Initial velocity in J2000 frame [km/s].
    epoch_dt : datetime
        Epoch of the initial state.
    period_s : float
        Orbital period [seconds].
    a_km : float
        Semi-major axis [km].
    ecc : float
        Eccentricity.
    cd_a_over_m : float
        Ballistic coefficient Cd*A/m [m^2/kg].

    Returns
    -------
    Path
        Path to the generated .script file.
    """
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epoch_str = _datetime_to_gmat_epoch(epoch_dt)
    prop_duration = prop_duration_override if prop_duration_override else N_ORBITS * period_s
    report_step = GMAT_STEP_SEC

    # Satellite area and mass from Cd*A/m
    # Assume Cd = 2.2 (standard), solve for A/m
    # Cd*A/m = cd_a_over_m => A/m = cd_a_over_m / 2.2
    # For the script, we need separate Cd, area, mass.
    # Use mass = 1000 kg (arbitrary, only ratio matters)
    cd = 2.2
    mass_kg = 1000.0
    area_m2 = cd_a_over_m * mass_kg / cd  # A = (Cd*A/m) * m / Cd
    cr = 1.8  # coefficient of reflectivity (typical)
    srp_area_m2 = area_m2  # assume same area for SRP

    report_path = OUTPUT_DIR / f"{norad_id}{script_suffix}_gmat_report.txt"

    script = f"""%----------------------------------------
% GMAT High-Fidelity Propagation Script
% Satellite: {name} (NORAD {norad_id})
% Generated by generate_gmat_data.py
%----------------------------------------

%--- Spacecraft ---
Create Spacecraft Sat;
GMAT Sat.DateFormat = UTCGregorian;
GMAT Sat.Epoch = '{epoch_str}';
GMAT Sat.CoordinateSystem = EarthMJ2000Eq;
GMAT Sat.DisplayStateType = Cartesian;
GMAT Sat.X = {pos_j2000[0]:.12f};
GMAT Sat.Y = {pos_j2000[1]:.12f};
GMAT Sat.Z = {pos_j2000[2]:.12f};
GMAT Sat.VX = {vel_j2000[0]:.12f};
GMAT Sat.VY = {vel_j2000[1]:.12f};
GMAT Sat.VZ = {vel_j2000[2]:.12f};
GMAT Sat.DryMass = {mass_kg:.1f};
GMAT Sat.Cd = {cd:.1f};
GMAT Sat.DragArea = {area_m2:.4f};
GMAT Sat.Cr = {cr:.1f};
GMAT Sat.SRPArea = {srp_area_m2:.4f};

%--- Force Model (high-fidelity) ---
Create ForceModel HighFidelityFM;
GMAT HighFidelityFM.CentralBody = Earth;
GMAT HighFidelityFM.PrimaryBodies = {{Earth}};
GMAT HighFidelityFM.PointMasses = {{Sun, Luna}};
GMAT HighFidelityFM.SRP = On;
GMAT HighFidelityFM.GravityField.Earth.Degree = 20;
GMAT HighFidelityFM.GravityField.Earth.Order = 20;
GMAT HighFidelityFM.GravityField.Earth.PotentialFile = 'JGM2.cof';
GMAT HighFidelityFM.Drag.AtmosphereModel = MSISE90;

%--- Propagator ---
Create Propagator HighFidelityProp;
GMAT HighFidelityProp.FM = HighFidelityFM;
GMAT HighFidelityProp.Type = RungeKutta89;
GMAT HighFidelityProp.InitialStepSize = 60;
GMAT HighFidelityProp.Accuracy = 1e-12;
GMAT HighFidelityProp.MinStep = 0.001;
GMAT HighFidelityProp.MaxStep = 300;
GMAT HighFidelityProp.MaxStepAttempts = 50;

%--- Report File ---
Create ReportFile OrbitReport;
GMAT OrbitReport.Filename = '{report_path.as_posix()}';
GMAT OrbitReport.Add = {{Sat.ElapsedSecs, Sat.EarthMJ2000Eq.X, Sat.EarthMJ2000Eq.Y, Sat.EarthMJ2000Eq.Z, Sat.EarthMJ2000Eq.VX, Sat.EarthMJ2000Eq.VY, Sat.EarthMJ2000Eq.VZ}};
GMAT OrbitReport.WriteHeaders = false;
GMAT OrbitReport.LeftJustify = On;
GMAT OrbitReport.ZeroFill = On;
GMAT OrbitReport.ColumnWidth = 23;

%--- Mission Sequence ---
BeginMissionSequence;
Propagate HighFidelityProp(Sat) {{Sat.ElapsedSecs = {prop_duration:.2f}}};
"""

    script_path = SCRIPT_DIR / f"{norad_id}{script_suffix}.script"
    script_path.write_text(script)

    return script_path


# ---------------------------------------------------------------------------
# GMAT execution and output parsing
# ---------------------------------------------------------------------------

def run_gmat_script(script_path: Path) -> bool:
    """Run a GMAT script via GmatConsole.

    Parameters
    ----------
    script_path : Path
        Path to the .script file.

    Returns
    -------
    bool
        True if GMAT exited successfully.
    """
    try:
        console = get_gmat_console_path()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False

    gmat_root = get_gmat_root()

    # R2022a: GMAT.exe --minimize --run --exit <script>
    # Older:  GmatConsole.exe --run --exit <script>
    is_gui_exe = console.name.lower() == "gmat.exe"
    cmd = [str(console)]
    if is_gui_exe:
        cmd += ["--minimize", "--no_splash"]
    cmd += ["--run", "--exit", str(script_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per satellite
            cwd=str(gmat_root),
        )
        if result.returncode != 0:
            print(f"  GMAT failed (exit code {result.returncode})")
            if result.stderr:
                # Print first few lines of stderr
                for line in result.stderr.strip().splitlines()[:5]:
                    print(f"    {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  GMAT timed out (>300s)")
        return False
    except Exception as e:
        print(f"  GMAT error: {e}")
        return False


def parse_gmat_report(report_path: Path) -> np.ndarray:
    """Parse a GMAT ReportFile into a numpy array.

    Expects columns: ElapsedSecs, X, Y, Z, VX, VY, VZ
    (7 numeric columns, whitespace-delimited, no headers).

    Returns
    -------
    np.ndarray, shape (N, 7)
        Columns: [t, x, y, z, vx, vy, vz]
        t in seconds, positions in km, velocities in km/s.
    """
    data = np.loadtxt(str(report_path))

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 7:
        raise ValueError(
            f"Expected 7 columns in GMAT report, got {data.shape[1]}"
        )

    return data


def resample_trajectory(data: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """Resample trajectory to exactly n_points evenly-spaced time steps.

    Uses cubic spline interpolation.

    Parameters
    ----------
    data : np.ndarray, shape (N, 7)
        Raw GMAT output [t, x, y, z, vx, vy, vz].
    n_points : int
        Number of output points.

    Returns
    -------
    np.ndarray, shape (n_points, 7)
        Resampled trajectory.
    """
    t_raw = data[:, 0]
    t_uniform = np.linspace(t_raw[0], t_raw[-1], n_points)

    result = np.zeros((n_points, 7), dtype=np.float64)
    result[:, 0] = t_uniform

    # Cubic spline interpolation for each state component
    for col in range(1, 7):
        cs = CubicSpline(t_raw, data[:, col])
        result[:, col] = cs(t_uniform)

    return result


# ---------------------------------------------------------------------------
# Main pipeline for a single satellite
# ---------------------------------------------------------------------------

def process_satellite(
    norad_id: int,
    name: str,
    cd_a_over_m: float = 0.022,
    dry_run: bool = False,
    long_arc: bool = False,
) -> bool:
    """Generate GMAT data for a single satellite.

    Parameters
    ----------
    norad_id : int
        NORAD catalog ID.
    name : str
        Satellite name.
    cd_a_over_m : float
        Ballistic coefficient Cd*A/m [m^2/kg].
    dry_run : bool
        If True, generate script but don't run GMAT.

    Returns
    -------
    bool
        True if successful (or dry_run completed).
    """
    # 1. Load TLE
    tle_result = load_tle(norad_id)
    if tle_result is None:
        print(f"  SKIPPED (no TLE for NORAD {norad_id})")
        return False

    line1, line2 = tle_result

    # 2. Get SGP4 state at epoch
    try:
        pos_teme, vel_teme, epoch_dt, period_s, a_km, inc_deg, ecc = \
            get_sgp4_state_at_epoch(line1, line2)
    except RuntimeError as e:
        print(f"  SKIPPED (SGP4 error: {e})")
        return False

    # 3. Convert TEME -> J2000
    pos_j2000, vel_j2000 = teme_to_j2000(pos_teme, vel_teme, epoch_dt)

    # 4. Generate GMAT script
    prop_dur = None
    script_suffix = ""
    n_out_points = N_POINTS
    if long_arc:
        prop_dur = LONG_ARC_DAYS * 86400.0
        script_suffix = "_7day"
        n_out_points = N_POINTS_LONG

    script_path = generate_gmat_script(
        norad_id=norad_id,
        name=name,
        pos_j2000=pos_j2000,
        vel_j2000=vel_j2000,
        epoch_dt=epoch_dt,
        period_s=period_s,
        a_km=a_km,
        ecc=ecc,
        cd_a_over_m=cd_a_over_m,
        prop_duration_override=prop_dur,
        script_suffix=script_suffix,
    )

    print(f"  Script: {script_path}")
    print(f"  Epoch:  {epoch_dt}")
    print(f"  J2000 pos: [{pos_j2000[0]:.3f}, {pos_j2000[1]:.3f}, {pos_j2000[2]:.3f}] km")
    print(f"  a={a_km:.1f} km, inc={inc_deg:.1f} deg, ecc={ecc:.4f}, T={period_s:.1f} s")

    if dry_run:
        print("  [DRY RUN] Script generated, skipping GMAT execution")
        return True

    # 5. Run GMAT
    print("  Running GMAT...", end="", flush=True)
    report_path = OUTPUT_DIR / f"{norad_id}{script_suffix}_gmat_report.txt"

    gmat_t0 = time.perf_counter()
    if not run_gmat_script(script_path):
        return False
    gmat_runtime_ms = (time.perf_counter() - gmat_t0) * 1e3

    # 6. Parse output
    if not report_path.exists():
        print(f"  ERROR: Report file not found: {report_path}")
        return False

    try:
        raw_data = parse_gmat_report(report_path)
    except (ValueError, Exception) as e:
        print(f"  ERROR parsing report: {e}")
        return False

    print(f" done ({raw_data.shape[0]} raw points)")

    # 7. Resample
    data = resample_trajectory(raw_data, n_out_points)

    # 8. Save .npy
    npy_path = OUTPUT_DIR / f"{norad_id}{script_suffix}.npy"
    np.save(str(npy_path), data)

    # 9. Save metadata
    n_orbits_actual = int(prop_dur / period_s) if prop_dur else N_ORBITS
    meta = {
        "norad_id": norad_id,
        "name": name,
        "a_km": float(a_km),
        "inc_deg": float(inc_deg),
        "ecc": float(ecc),
        "period_s": float(period_s),
        "n_orbits": n_orbits_actual,
        "n_points": n_out_points,
        "data_source": "GMAT",
        "force_model": {
            "gravity": "JGM-2 20x20",
            "drag": "MSISE-90",
            "srp": True,
            "third_body": ["Sun", "Luna"],
            "integrator": "RungeKutta89",
            "accuracy": 1e-12,
        },
        "epoch_utc": epoch_dt.isoformat(),
        "frame": "EarthMJ2000Eq",
        "cd_a_over_m": cd_a_over_m,
        "raw_gmat_points": int(raw_data.shape[0]),
        "gmat_runtime_ms": round(gmat_runtime_ms, 1),
    }
    meta_path = OUTPUT_DIR / f"{norad_id}{script_suffix}_meta.json" if script_suffix else OUTPUT_DIR / f"{norad_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # 10. Clean up raw report (large file)
    if report_path.exists():
        report_path.unlink()

    t_hours = data[-1, 0] / 3600.0
    r = np.linalg.norm(data[:, 1:4], axis=1)
    print(f"  Saved: {npy_path}")
    print(f"  span={t_hours:.1f}h, r=[{r.min():.1f}, {r.max():.1f}] km")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-fidelity trajectory data using NASA GMAT"
    )
    parser.add_argument(
        "--sat", type=int, default=None,
        help="NORAD ID of a single satellite (default: full catalog)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate GMAT scripts without running them"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate GMAT installation and exit"
    )
    parser.add_argument(
        "--long-arc", action="store_true",
        help="Generate 7-day trajectories (default: 5-orbit)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  GMAT High-Fidelity Trajectory Generation")
    print("=" * 70)

    # -- Validate mode --
    if args.validate:
        print("\n  Validating GMAT installation...")
        ok = validate_gmat_install()
        sys.exit(0 if ok else 1)

    # -- Build satellite list --
    if args.sat is not None:
        entry = get_by_norad_id(args.sat)
        if entry is None:
            print(f"\n  ERROR: NORAD ID {args.sat} not found in catalog.")
            sys.exit(1)
        satellites = [entry]
    else:
        satellites = get_catalog()

    print(f"\n  Satellites: {len(satellites)}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Mode:       {'7-day long-arc' if args.long_arc else '5-orbit'}")
    print(f"  Dry run:    {args.dry_run}")
    if not args.dry_run:
        try:
            console = get_gmat_console_path()
            print(f"  GMAT:       {console}")
        except FileNotFoundError as e:
            print(f"\n  {e}")
            print("\n  Use --dry-run to generate scripts without GMAT.")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_fail = 0

    for idx, sat in enumerate(satellites):
        print(f"\n{'=' * 70}")
        print(f"  [{idx + 1}/{len(satellites)}] {sat.name} (NORAD {sat.norad_id})")
        print(f"{'=' * 70}")

        ok = process_satellite(
            norad_id=sat.norad_id,
            name=sat.name,
            cd_a_over_m=sat.cd_a_over_m,
            dry_run=args.dry_run,
            long_arc=args.long_arc,
        )

        if ok:
            n_success += 1
        else:
            n_fail += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Succeeded: {n_success}/{len(satellites)}")
    print(f"  Failed:    {n_fail}/{len(satellites)}")
    if args.dry_run:
        print(f"  Scripts in: {SCRIPT_DIR}")
    else:
        print(f"  Data in:    {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
