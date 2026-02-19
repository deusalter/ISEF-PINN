"""
download_tle.py
===============
Download Two-Line Element sets (TLEs) from Space-Track for satellites in the
project catalog.  Falls back to hardcoded TLEs when credentials are unavailable
(offline development).

Usage:
    python download_tle.py

Outputs:
    data/tle_catalog/{norad_id}.tle   -- two-line TLE text
    data/tle_catalog/{norad_id}.json  -- structured dict (line1, line2, epoch, norad_id, name)
"""

import os
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so it works from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TLE_DIR = SCRIPT_DIR / "data" / "tle_catalog"
ENV_PATH = SCRIPT_DIR / ".env"

sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Hardcoded fallback TLEs for offline development
# ---------------------------------------------------------------------------
FALLBACK_TLES: dict[int, dict] = {
    # -- Low-inclination --
    44883: {
        "name": "CBERS 04A",
        "line1": "1 44883U 20003A   24001.50000000  .00000200  00000-0  28000-4 0  9996",
        "line2": "2 44883  28.5000 120.0000 0001000  45.0000 315.0000 14.81500000200000",
    },
    57320: {
        "name": "TROPICS-01",
        "line1": "1 57320U 23084A   24001.50000000  .00005000  00000-0  32000-3 0  9991",
        "line2": "2 57320  29.7000 200.0000 0010000  60.0000 300.0000 15.06000000 50000",
    },
    57321: {
        "name": "TROPICS-02",
        "line1": "1 57321U 23084B   24001.50000000  .00005000  00000-0  32000-3 0  9992",
        "line2": "2 57321  29.7000 200.5000 0010000  62.0000 298.0000 15.06000000 50000",
    },
    25063: {
        "name": "ORBCOMM FM-5",
        "line1": "1 25063U 97084B   24001.50000000  .00000100  00000-0  52000-4 0  9999",
        "line2": "2 25063  25.0000  90.0000 0010000 100.0000 260.0000 14.33000000300000",
    },
    # -- ISS-like --
    25544: {
        "name": "ISS (ZARYA)",
        "line1": "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
        "line2": "2 25544  51.6400 208.5000 0006000  30.0000 330.0000 15.50000000400000",
    },
    48274: {
        "name": "TIANHE (CSS)",
        "line1": "1 48274U 21035A   24001.50000000  .00020000  00000-0  14000-3 0  9993",
        "line2": "2 48274  41.4700 180.0000 0003000  40.0000 320.0000 15.60000000150000",
    },
    56227: {
        "name": "CYGNUS NG-19",
        "line1": "1 56227U 23039A   24001.50000000  .00015000  00000-0  95000-4 0  9994",
        "line2": "2 56227  51.6300 210.0000 0003000  35.0000 325.0000 15.50000000 80000",
    },
    58536: {
        "name": "CREW DRAGON C212",
        "line1": "1 58536U 23185A   24001.50000000  .00015000  00000-0  95000-4 0  9995",
        "line2": "2 58536  51.6400 215.0000 0002000  25.0000 335.0000 15.50000000 30000",
    },
    # -- Constellation / Starlink --
    44713: {
        "name": "STARLINK-1007",
        "line1": "1 44713U 19074A   24001.50000000  .00001264  00000-0  93842-4 0  9991",
        "line2": "2 44713  53.0554 170.2100 0001500  80.0000 280.1000 15.06400000230000",
    },
    44714: {
        "name": "STARLINK-1008",
        "line1": "1 44714U 19074B   24001.50000000  .00001300  00000-0  96000-4 0  9992",
        "line2": "2 44714  53.0500 170.5000 0001400  82.0000 278.0000 15.06400000230000",
    },
    44715: {
        "name": "STARLINK-1009",
        "line1": "1 44715U 19074C   24001.50000000  .00001250  00000-0  92000-4 0  9993",
        "line2": "2 44715  53.0600 171.0000 0001300  84.0000 276.0000 15.06400000230000",
    },
    44716: {
        "name": "STARLINK-1010",
        "line1": "1 44716U 19074D   24001.50000000  .00001280  00000-0  94000-4 0  9994",
        "line2": "2 44716  53.0500 171.5000 0001500  86.0000 274.0000 15.06400000230000",
    },
    # -- Sun-synchronous --
    49260: {
        "name": "LANDSAT 9",
        "line1": "1 49260U 21088A   24001.50000000  .00000400  00000-0  94000-4 0  9997",
        "line2": "2 49260  98.2100  75.0000 0001200  90.0000 270.1000 14.57100000120000",
    },
    46984: {
        "name": "SENTINEL-6A",
        "line1": "1 46984U 20084A   24001.50000000  .00000100  00000-0  30000-4 0  9998",
        "line2": "2 46984  66.0400  50.0000 0005000 110.0000 250.0000 14.27000000160000",
    },
    43013: {
        "name": "NOAA-20 (JPSS-1)",
        "line1": "1 43013U 17073A   24001.50000000  .00000200  00000-0  48000-4 0  9996",
        "line2": "2 43013  98.7200  20.0000 0001000  70.0000 290.0000 14.19500000330000",
    },
    37849: {
        "name": "SUOMI NPP",
        "line1": "1 37849U 11061A   24001.50000000  .00000200  00000-0  49000-4 0  9997",
        "line2": "2 37849  98.7300  22.0000 0001000  68.0000 292.0000 14.19500000650000",
    },
    # -- Diverse --
    20580: {
        "name": "HST",
        "line1": "1 20580U 90037B   24001.50000000  .00000800  00000-0  39000-4 0  9994",
        "line2": "2 20580  28.4700 140.0000 0002800 200.0000 160.0000 15.09100000420000",
    },
    43476: {
        "name": "GRACE-FO 1",
        "line1": "1 43476U 18047A   24001.50000000  .00000600  00000-0  26000-4 0  9998",
        "line2": "2 43476  89.0100  60.0000 0010000 120.0000 240.0000 15.18000000300000",
    },
    43070: {
        "name": "IRIDIUM 106",
        "line1": "1 43070U 17083F   24001.50000000  .00000080  00000-0  20000-4 0  9999",
        "line2": "2 43070  86.3900  40.0000 0002000 130.0000 230.0000 14.34200000330000",
    },
    36508: {
        "name": "COSMOS 2251 DEB",
        "line1": "1 36508U 93036PJ  24001.50000000  .00000400  00000-0  14000-3 0  9997",
        "line2": "2 36508  74.0400  80.0000 0100000 150.0000 210.0000 14.12000000750000",
    },
}


def _parse_epoch_from_tle(line1: str) -> str:
    """Extract the epoch string (YYDDD.DDDDDDDD) from TLE line 1."""
    # Columns 18-32 of line 1 contain the epoch
    return line1[18:32].strip()


def _parse_tle_pairs(raw_text: str) -> list[tuple[str, str]]:
    """Parse raw TLE text into a list of (line1, line2) tuples."""
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    pairs = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            pairs.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return pairs


def _norad_id_from_line1(line1: str) -> int:
    """Extract NORAD catalog ID from TLE line 1 (columns 2-7)."""
    return int(line1[2:7].strip())


def _load_credentials() -> tuple[str | None, str | None]:
    """Load Space-Track credentials from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        # python-dotenv not installed; fall through to os.environ
        pass

    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    return user, password


def _save_tle(norad_id: int, name: str, line1: str, line2: str) -> None:
    """Save a TLE to disk as both .tle and .json files."""
    TLE_DIR.mkdir(parents=True, exist_ok=True)

    # .tle file (two-line format)
    tle_path = TLE_DIR / f"{norad_id}.tle"
    tle_path.write_text(f"{line1}\n{line2}\n")

    # .json file (structured dict)
    meta = {
        "norad_id": norad_id,
        "name": name,
        "line1": line1,
        "line2": line2,
        "epoch": _parse_epoch_from_tle(line1),
    }
    json_path = TLE_DIR / f"{norad_id}.json"
    json_path.write_text(json.dumps(meta, indent=2) + "\n")


def load_tle(norad_id: int) -> tuple[str, str] | None:
    """
    Read a saved TLE from disk.

    Parameters
    ----------
    norad_id : int
        NORAD catalog number.

    Returns
    -------
    (line1, line2) or None if the file does not exist.
    """
    json_path = TLE_DIR / f"{norad_id}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return (data["line1"], data["line2"])

    tle_path = TLE_DIR / f"{norad_id}.tle"
    if tle_path.exists():
        lines = tle_path.read_text().strip().splitlines()
        if len(lines) >= 2:
            return (lines[0].strip(), lines[1].strip())

    return None


def download_all_tles() -> dict[int, tuple[str, str]]:
    """
    Download TLEs for every satellite in the catalog.

    Tries Space-Track first; falls back to hardcoded TLEs if credentials are
    missing or the download fails.

    Returns
    -------
    dict mapping norad_id -> (line1, line2)
    """
    # ------------------------------------------------------------------
    # Build the list of NORAD IDs from the satellite catalog
    # ------------------------------------------------------------------
    try:
        from satellite_catalog import get_catalog
        catalog = get_catalog()  # expected: list of dicts with at least "norad_id" and "name"
    except (ImportError, ModuleNotFoundError):
        print("Warning: satellite_catalog.py not found. Using fallback satellite list.")
        catalog = [
            {"norad_id": nid, "name": info["name"]}
            for nid, info in FALLBACK_TLES.items()
        ]

    # catalog may be a list of dicts or SatelliteEntry dataclass instances
    def _get(sat, key, default=None):
        if isinstance(sat, dict):
            return sat.get(key, default)
        return getattr(sat, key, default)

    norad_ids = [_get(sat, "norad_id") for sat in catalog]
    name_map = {_get(sat, "norad_id"): _get(sat, "name", f"NORAD {_get(sat, 'norad_id')}") for sat in catalog}

    print(f"Catalog contains {len(norad_ids)} satellites: {norad_ids}")

    # ------------------------------------------------------------------
    # Attempt Space-Track download
    # ------------------------------------------------------------------
    results: dict[int, tuple[str, str]] = {}
    downloaded_from_api = False

    user, password = _load_credentials()
    if user and password:
        print("\nSpace-Track credentials found. Downloading TLEs...")
        try:
            from spacetrack import SpaceTrackClient

            st = SpaceTrackClient(identity=user, password=password)
            raw = st.tle_latest(
                norad_cat_id=norad_ids,
                ordinal=1,
                format="tle",
            )

            pairs = _parse_tle_pairs(raw)
            for line1, line2 in pairs:
                nid = _norad_id_from_line1(line1)
                results[nid] = (line1, line2)
                name = name_map.get(nid, f"NORAD {nid}")
                _save_tle(nid, name, line1, line2)
                print(f"  [OK] {nid} ({name})")

            downloaded_from_api = True

        except ImportError:
            print("Warning: 'spacetrack' library not installed. Falling back to hardcoded TLEs.")
        except Exception as exc:
            print(f"Warning: Space-Track download failed: {exc}")
            print("Falling back to hardcoded TLEs.")
    else:
        print("\nNo Space-Track credentials found (set SPACETRACK_USER and SPACETRACK_PASS in .env).")
        print("Using hardcoded fallback TLEs.")

    # ------------------------------------------------------------------
    # Fill in any satellites that were not downloaded from Space-Track
    # ------------------------------------------------------------------
    missing = [nid for nid in norad_ids if nid not in results]
    if missing:
        if downloaded_from_api:
            print(f"\n{len(missing)} satellite(s) missing from Space-Track response. "
                  f"Checking fallbacks...")
        for nid in missing:
            if nid in FALLBACK_TLES:
                fb = FALLBACK_TLES[nid]
                line1, line2 = fb["line1"], fb["line2"]
                name = name_map.get(nid, fb["name"])
                results[nid] = (line1, line2)
                _save_tle(nid, name, line1, line2)
                print(f"  [FALLBACK] {nid} ({name})")
            else:
                print(f"  [MISSING]  {nid} ({name_map.get(nid, '?')}) -- no TLE available")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_ok = len(results)
    n_fail = len(norad_ids) - n_ok
    print(f"\nSummary: {n_ok} succeeded, {n_fail} failed out of {len(norad_ids)} satellites.")
    if n_ok > 0:
        print(f"TLEs saved to: {TLE_DIR}/")

    return results


def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("  TLE Downloader for ISEF PINN Orbital Propagation Project")
    print("=" * 60)
    results = download_all_tles()

    if results:
        print("\nDownloaded TLEs:")
        for nid, (l1, l2) in sorted(results.items()):
            epoch = _parse_epoch_from_tle(l1)
            print(f"  {nid:>6d}  epoch={epoch}  {l1[:24]}...")
    else:
        print("\nNo TLEs were obtained.")


if __name__ == "__main__":
    main()
