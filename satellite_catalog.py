"""
Curated catalog of ~20 diverse LEO satellites for PINN orbital propagation research.

Each entry contains the NORAD catalog ID, common name, orbit classification,
and approximate orbital elements (altitude, inclination, eccentricity).
All satellites are in low Earth orbit (alt < 900 km) with near-circular
orbits (ecc < 0.02).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SatelliteEntry:
    norad_id: int
    name: str
    orbit_type: str
    approx_alt_km: float
    approx_inc_deg: float
    approx_ecc: float


# ---------------------------------------------------------------------------
# Low-inclination (inc < 30 deg)
# ---------------------------------------------------------------------------
_LOW_INC = [
    SatelliteEntry(44883, "CBERS 04A",          "low-inclination", 628.0, 28.5, 0.0001),
    SatelliteEntry(57320, "TROPICS-01",         "low-inclination", 550.0, 29.7, 0.0010),
    SatelliteEntry(57321, "TROPICS-02",         "low-inclination", 550.0, 29.7, 0.0010),
    SatelliteEntry(25063, "ORBCOMM FM-5",       "low-inclination", 775.0, 25.0, 0.0010),
]

# ---------------------------------------------------------------------------
# ISS-like (~51.6 deg inclination)
# ---------------------------------------------------------------------------
_ISS_LIKE = [
    SatelliteEntry(25544, "ISS (ZARYA)",         "ISS-like", 420.0, 51.6, 0.0002),
    SatelliteEntry(48274, "TIANHE (CSS)",         "ISS-like", 390.0, 41.5, 0.0003),
    SatelliteEntry(56227, "CYGNUS NG-19",        "ISS-like", 415.0, 51.6, 0.0003),
    SatelliteEntry(58536, "CREW DRAGON C212",    "ISS-like", 420.0, 51.6, 0.0002),
]

# ---------------------------------------------------------------------------
# Constellation / Starlink (~550 km, ~53 deg)
# ---------------------------------------------------------------------------
_STARLINK = [
    SatelliteEntry(44713, "STARLINK-1007",       "constellation", 550.0, 53.0, 0.0001),
    SatelliteEntry(44714, "STARLINK-1008",       "constellation", 550.0, 53.0, 0.0001),
    SatelliteEntry(44715, "STARLINK-1009",       "constellation", 550.0, 53.0, 0.0001),
    SatelliteEntry(44716, "STARLINK-1010",       "constellation", 550.0, 53.0, 0.0001),
]

# ---------------------------------------------------------------------------
# Sun-synchronous (~97-98 deg, 700-830 km)
# ---------------------------------------------------------------------------
_SSO = [
    SatelliteEntry(49260, "LANDSAT 9",           "sun-synchronous", 705.0, 98.2, 0.0001),
    SatelliteEntry(46984, "SENTINEL-6A",         "sun-synchronous", 830.0, 97.8, 0.0005),
    SatelliteEntry(43013, "NOAA-20 (JPSS-1)",    "sun-synchronous", 824.0, 98.7, 0.0001),
    SatelliteEntry(37849, "SUOMI NPP",           "sun-synchronous", 824.0, 98.7, 0.0001),
]

# ---------------------------------------------------------------------------
# Diverse / Other (mixed altitudes and inclinations)
# ---------------------------------------------------------------------------
_DIVERSE = [
    SatelliteEntry(20580, "HUBBLE (HST)",        "diverse", 540.0, 28.5, 0.0003),
    SatelliteEntry(43476, "GRACE-FO 1",          "diverse", 490.0, 89.0, 0.0010),
    SatelliteEntry(43070, "IRIDIUM 106",         "diverse", 780.0, 86.4, 0.0002),
    SatelliteEntry(36508, "COSMOS 2251 DEB",     "diverse", 850.0, 74.0, 0.0100),
]


CATALOG: List[SatelliteEntry] = _LOW_INC + _ISS_LIKE + _STARLINK + _SSO + _DIVERSE


def get_catalog() -> List[SatelliteEntry]:
    """Return the full satellite catalog."""
    return CATALOG


def get_by_norad_id(norad_id: int) -> Optional[SatelliteEntry]:
    """Return a single catalog entry by NORAD ID, or None if not found."""
    for entry in CATALOG:
        if entry.norad_id == norad_id:
            return entry
    return None
