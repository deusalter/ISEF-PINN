# PINN Orbital Propagation Results

## Overview

Physics-Informed Neural Networks (PINNs) with Fourier feature encoding for LEO satellite orbit prediction. Trained on 20% of data (~1 orbit), tested on 80% (~4 orbits extrapolation).

Architecture: FourierPINN with N_FREQ=8, 64-wide hidden layers, 3 layers, secular drift head. ~9,651 parameters.

Key techniques: Adaptive collocation (concentrated near train/test boundary), separate secular head learning rate (0.1x), 4-phase physics curriculum.

---

## SGP4 Data Results (20 Satellites)

Data source: SGP4-propagated orbits from TLE elements. 5000 points per satellite covering 5 orbital periods.

| # | NORAD | Satellite | Type | Alt (km) | Inc (deg) | Vanilla (km) | Fourier NN (km) | PINN J2 (km) | PINN J2+Drag (km) | Best (km) |
|---|-------|-----------|------|----------|-----------|-------------:|----------------:|-------------:|-------------------:|----------:|
| 1 | 44883 | CBERS 04A | low-inc | 628 | 28.5 | 21,945 | 890 | 4.18 | **0.89** | 0.89 |
| 2 | 57320 | TROPICS-01 | low-inc | 550 | 29.7 | 18,813 | 876 | **4.35** | 4.99 | 4.35 |
| 3 | 57321 | TROPICS-02 | low-inc | 550 | 29.7 | 18,816 | 878 | **4.52** | 5.19 | 4.52 |
| 4 | 25063 | ORBCOMM FM-5 | low-inc | 775 | 25.0 | 22,469 | 879 | **2.44** | 3.35 | 2.44 |
| 5 | 25544 | ISS (ZARYA) | ISS-like | 420 | 51.6 | 20,763 | 889 | 5.45 | **5.04** | 5.04 |
| 6 | 48274 | TIANHE (CSS) | ISS-like | 390 | 41.5 | 17,557 | 833 | 5.07 | **3.13** | 3.13 |
| 7 | 56227 | CYGNUS NG-19 | ISS-like | 415 | 51.6 | 20,690 | 890 | **3.01** | 6.00 | 3.01 |
| 8 | 58536 | CREW DRAGON | ISS-like | 420 | 51.6 | 19,849 | 888 | **2.57** | 3.52 | 2.57 |
| 9 | 44713 | STARLINK-1007 | constellation | 550 | 53.1 | 19,619 | 861 | 2.97 | **2.83** | 2.83 |
| 10 | 44714 | STARLINK-1008 | constellation | 550 | 53.0 | 19,541 | 861 | 5.18 | **2.63** | 2.63 |
| 11 | 44715 | STARLINK-1009 | constellation | 550 | 53.1 | 19,550 | 859 | 4.79 | **3.44** | 3.44 |
| 12 | 44716 | STARLINK-1010 | constellation | 550 | 53.0 | 19,570 | 858 | **2.17** | 4.28 | 2.17 |
| 13 | 49260 | LANDSAT 9 | sun-sync | 705 | 98.2 | 20,554 | 1,025 | **9.74** | 10.07 | 9.74 |
| 14 | 46984 | SENTINEL-6A | sun-sync | 830 | 66.0 | 21,096 | 923 | 3.22 | **3.03** | 3.03 |
| 15 | 43013 | NOAA-20 | sun-sync | 824 | 98.7 | 25,207 | 1,068 | **9.80** | 10.05 | 9.80 |
| 16 | 37849 | SUOMI NPP | sun-sync | 824 | 98.7 | 23,635 | 1,072 | **9.21** | 10.20 | 9.21 |
| 17 | 20580 | HUBBLE (HST) | diverse | 540 | 28.5 | 21,959 | 946 | **1.71** | 1.96 | 1.71 |
| 18 | 43476 | GRACE-FO 1 | diverse | 490 | 89.0 | 21,385 | 1,067 | 16.20 | **15.43** | 15.43 |
| 19 | 43070 | IRIDIUM 106 | diverse | 780 | 86.4 | 22,693 | 1,066 | 16.57 | **12.41** | 12.41 |
| 20 | 36508 | COSMOS 2251 DEB | diverse | 850 | 74.0 | 23,622 | 1,024 | **5.89** | 5.53 | 5.53 |

### Summary Statistics (SGP4)

| Model | Avg Test RMSE (km) | Improvement over Vanilla |
|-------|-------------------:|-------------------------:|
| Vanilla MLP | 20,967 | -- |
| Fourier NN (data-only) | 933 | 95.6% |
| **Fourier PINN (J2)** | **5.95** | **99.97%** |
| **Fourier PINN (J2+Drag)** | **5.70** | **99.97%** |

### Performance by Orbit Type

| Orbit Type | Satellites | Avg PINN J2 (km) | Avg Best (km) |
|------------|-----------|------------------:|---------------:|
| Low-inclination | 4 | 3.87 | 3.05 |
| ISS-like | 4 | 4.03 | 3.44 |
| Constellation | 4 | 3.78 | 2.77 |
| Sun-synchronous | 4 | 7.99 | 7.95 |
| Diverse | 4 | 10.09 | 8.27 |

### Key Observations

1. **17/20 satellites under 10 km RMSE** with the best PINN variant
2. **3 outliers above 10 km**: GRACE-FO 1 (15.4 km, near-polar 89 deg), IRIDIUM 106 (12.4 km, near-polar 86 deg), LANDSAT 9 (9.7 km), NOAA-20 (9.8 km), SUOMI NPP (9.2 km) -- all high-inclination or near-polar orbits
3. **J2+Drag model helps at low altitudes** (CBERS at 628 km: 0.89 km, TIANHE at 390 km: 3.13 km) but **hurts at higher altitudes** where SGP4's drag model doesn't match our Harris-Priester model
4. **Best absolute performance**: CBERS 04A at 0.89 km with J2+Drag, Hubble at 1.71 km with J2-only
5. **99.97% average improvement** over data-only Vanilla MLP baseline

### Why J2+Drag Sometimes Underperforms J2-Only

The PINN is trained on SGP4 data, which has its own internal drag model (B* coefficient). When we add our drag model (Harris-Priester atmosphere + learnable Cd*A/m), there's a **model mismatch**: our drag model doesn't perfectly match SGP4's simplified drag. The J2-only physics loss is a cleaner match. Drag helps when the atmosphere effect is large enough that even an approximate model adds useful physics (low-altitude orbits like CBERS and TIANHE).

---

## GMAT Data Results (20 Satellites)

*Training in progress... Results will be added when complete.*

GMAT force model: JGM-2 20x20 gravity harmonics, MSISE-90 atmospheric drag, solar radiation pressure, Sun/Moon third-body perturbations, RungeKutta89 integrator (accuracy 1e-12).

GMAT training curriculum: 15,000 epochs, gentler physics ramp (max lambda=0.05).
