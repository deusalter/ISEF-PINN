"""
Curated catalog of ~20 diverse LEO satellites for PINN orbital propagation research.

Each entry contains the NORAD catalog ID, common name, orbit classification,
approximate orbital elements (altitude, inclination, eccentricity), and the
ballistic drag coefficient cd_a_over_m = Cd * A / m  (m^2/kg).  This term
drives the atmospheric drag perturbation used by the J2+Drag PINN model.

Typical LEO values range from ~0.010 (heavy science spacecraft) to ~0.040
(tumbling debris).  Default is 0.022 m^2/kg.
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
    cd_a_over_m: float = 0.022  # Cd*A/m in m^2/kg (default typical LEO)


# ---------------------------------------------------------------------------
# Low-inclination (inc < 30 deg)
# ---------------------------------------------------------------------------
_LOW_INC = [
    SatelliteEntry(44883, "CBERS 04A",          "low-inclination", 628.0, 28.5, 0.0001, cd_a_over_m=0.015),
    SatelliteEntry(57320, "TROPICS-01",         "low-inclination", 550.0, 29.7, 0.0010, cd_a_over_m=0.030),
    SatelliteEntry(57321, "TROPICS-02",         "low-inclination", 550.0, 29.7, 0.0010, cd_a_over_m=0.030),
    SatelliteEntry(25063, "ORBCOMM FM-5",       "low-inclination", 775.0, 25.0, 0.0010, cd_a_over_m=0.025),
]

# ---------------------------------------------------------------------------
# ISS-like (~51.6 deg inclination)
# ---------------------------------------------------------------------------
_ISS_LIKE = [
    SatelliteEntry(25544, "ISS (ZARYA)",         "ISS-like", 420.0, 51.6, 0.0002, cd_a_over_m=0.020),
    SatelliteEntry(48274, "TIANHE (CSS)",         "ISS-like", 390.0, 41.5, 0.0003, cd_a_over_m=0.018),
    SatelliteEntry(56227, "CYGNUS NG-19",        "ISS-like", 415.0, 51.6, 0.0003, cd_a_over_m=0.025),
    SatelliteEntry(58536, "CREW DRAGON C212",    "ISS-like", 420.0, 51.6, 0.0002, cd_a_over_m=0.020),
]

# ---------------------------------------------------------------------------
# Constellation / Starlink (~550 km, ~53 deg)
# ---------------------------------------------------------------------------
_STARLINK = [
    SatelliteEntry(44713, "STARLINK-1007",       "constellation", 550.0, 53.0, 0.0001, cd_a_over_m=0.025),
    SatelliteEntry(44714, "STARLINK-1008",       "constellation", 550.0, 53.0, 0.0001, cd_a_over_m=0.025),
    SatelliteEntry(44715, "STARLINK-1009",       "constellation", 550.0, 53.0, 0.0001, cd_a_over_m=0.025),
    SatelliteEntry(44716, "STARLINK-1010",       "constellation", 550.0, 53.0, 0.0001, cd_a_over_m=0.025),
]

# ---------------------------------------------------------------------------
# Sun-synchronous (~97-98 deg, 700-830 km)
# ---------------------------------------------------------------------------
_SSO = [
    SatelliteEntry(49260, "LANDSAT 9",           "sun-synchronous", 705.0, 98.2, 0.0001, cd_a_over_m=0.015),
    SatelliteEntry(46984, "SENTINEL-6A",         "sun-synchronous", 830.0, 97.8, 0.0005, cd_a_over_m=0.012),
    SatelliteEntry(43013, "NOAA-20 (JPSS-1)",    "sun-synchronous", 824.0, 98.7, 0.0001, cd_a_over_m=0.015),
    SatelliteEntry(37849, "SUOMI NPP",           "sun-synchronous", 824.0, 98.7, 0.0001, cd_a_over_m=0.015),
]

# ---------------------------------------------------------------------------
# Diverse / Other (mixed altitudes and inclinations)
# ---------------------------------------------------------------------------
_DIVERSE = [
    SatelliteEntry(20580, "HUBBLE (HST)",        "diverse", 540.0, 28.5, 0.0003, cd_a_over_m=0.010),
    SatelliteEntry(43476, "GRACE-FO 1",          "diverse", 490.0, 89.0, 0.0010, cd_a_over_m=0.020),
    SatelliteEntry(43070, "IRIDIUM 106",         "diverse", 780.0, 86.4, 0.0002, cd_a_over_m=0.022),
    SatelliteEntry(36508, "COSMOS 2251 DEB",     "diverse", 850.0, 74.0, 0.0100, cd_a_over_m=0.040),
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
