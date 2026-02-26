# Physics-Informed Neural Networks for High-Fidelity Orbital Propagation and Real-Time LEO Conjunction Assessment

## Abstract

This project develops a Physics-Informed Neural Network (PINN) framework for propagating low-Earth-orbit (LEO) satellite trajectories by embedding gravitational dynamics directly into the neural network training loss via PyTorch automatic differentiation. A Fourier feature encoding layer provides the network with natural periodicity building blocks (sin/cos at orbital-frequency harmonics), enabling robust extrapolation of orbital dynamics. Unlike conventional data-driven networks that treat orbital mechanics as a black box, the PINN enforces the governing equations of motion at a dense set of collocation points spanning the full time domain -- including regions where labelled data are absent.

To ensure a rigorous, non-circular comparison, we use NASA's General Mission Analysis Tool (GMAT) as independent high-fidelity ground truth (20x20 JGM-2 gravity, MSISE-90 drag, SRP, Sun/Moon perturbations, RK89 integrator at 1e-12 accuracy). The PINN and SGP4 are each compared against GMAT, avoiding the methodological flaw of training on SGP4 data and then comparing against SGP4.

Key results across 20 real LEO satellites (5 orbit types):
- **99.97% average RMSE reduction** over vanilla MLPs (20,967 km -> 5.70 km)
- **17/20 satellites under 10 km RMSE** with best PINN variant
- **98.2% test RMSE reduction** on J2-perturbed extrapolation (1 orbit training, 4 orbit prediction)
- Atmospheric drag provides marginal improvement over J2-only physics for short-arc propagation

---

## Step-by-Step Reproduction Guide

This section is written so that anyone -- including ISEF judges -- can replicate every number in this project from scratch on a fresh machine. Follow the steps in order.

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or later | 3.11 and 3.12 also tested |
| pip | any recent | Comes with Python |
| Git | any recent | For cloning the repo |
| NASA GMAT | R2022a | Optional for GMAT ground truth; SGP4 pipeline works without it |

**Operating system:** Windows 10/11, macOS, or Linux. All scripts are cross-platform.
**GPU:** Not required. All training runs on CPU using `float64` precision.

### Step 0: Clone the Repository and Install Dependencies

```bash
git clone https://github.com/deusalter/ISEF-PINN.git
cd ISEF-PINN

# Create a virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

The `requirements.txt` installs: numpy, scipy, matplotlib, torch, sgp4, spacetrack, python-dotenv, astropy.

### Step 1: Generate Synthetic Ground Truth Data

This creates high-precision reference trajectories by numerically integrating the equations of motion using SciPy's DOP853 integrator (8th-order Runge-Kutta, rtol = atol = 1e-12).

```bash
python generate_data.py
```

**What it produces:**
- `data/orbital_data.npy` -- Two-body (Keplerian) trajectory, shape (2000, 7): `[t, x, y, z, vx, vy, vz]`
- `data/orbital_data_j2.npy` -- J2-perturbed trajectory, shape (5000, 7)

**Runtime:** < 10 seconds

### Step 2: Train the Vanilla MLP Baseline

```bash
python train_baseline.py
```

**What it produces:**
- `data/vanilla_predictions.npy` -- Vanilla MLP position predictions
- `data/vanilla_loss_history.npy` -- Per-epoch training loss
- `models/vanilla_mlp.pt` -- Saved model weights

**Runtime:** 1--3 minutes (5,000 epochs)

### Step 3: Train the Two-Body PINN

```bash
python train_pinn.py
```

**What it produces:**
- `data/pinn_predictions.npy` -- PINN position predictions
- `data/pinn_loss_history.npy` -- Per-epoch [total, data, physics] losses
- `models/pinn_twobody.pt` -- Saved model weights

**Runtime:** 10--15 minutes (10,000 epochs with physics curriculum)

### Step 4: Train the J2-Perturbed Fourier-PINN

```bash
python train_pinn_j2.py
```

Trains both a matched vanilla MLP baseline and the J2-physics-informed Fourier-PINN on a 400 km inclined LEO orbit with Earth oblateness (J2 = 1.08263e-3). Uses 20% training / 80% test split (1 orbit training, 4 orbit prediction).

**What it produces:**
- `data/pinn_j2_predictions.npy`, `data/vanilla_j2_predictions.npy`
- `models/pinn_j2.pt`
- Figures in `figures/j2_*.png`

**Runtime:** 15--20 minutes

### Step 5: Evaluate and Generate All Figures

```bash
python evaluate.py
```

Loads all saved predictions from Steps 2--4, computes RMSE metrics, and generates publication-quality figures.

**What it produces:** All figures in `figures/` (see Output Figures section below).

**Runtime:** < 60 seconds

### Step 6: Real Satellite Catalog Validation (SGP4 Data)

This trains and evaluates the PINN on 20 real LEO satellites from the NORAD catalog.

```bash
# 6a. Download Two-Line Element sets (TLEs)
#     Uses hardcoded fallback TLEs if no Space-Track credentials are available.
#     No account needed -- fallbacks are included in the repo.
python download_tle.py

# 6b. Propagate TLEs into dense trajectories using SGP4
python generate_real_data.py

# 6c. Train 4 models per satellite across all 20 satellites
python train_real_orbits.py

# 6d. Generate statistical analysis and publication figures
python analyze_catalog.py
```

**To test on a single satellite first (recommended):**

```bash
python train_real_orbits.py --sat 25544   # ISS only, ~90 seconds
```

**What it produces:**
- `data/real_orbits/{norad_id}.npy` -- SGP4-propagated trajectories (5000 points each)
- `data/real_orbits/{norad_id}_meta.json` -- Orbital metadata
- `data/real_results/{norad_id}_results.json` -- Per-satellite results (4 models each)
- `data/real_results/summary.json` -- Aggregate results across all 20 satellites
- `figures/catalog_*.png` -- Publication figures

**Runtime:** ~60 minutes for all 20 satellites (4 models x 10,000 epochs each)

### Step 7: GMAT High-Fidelity Ground Truth (Independent Validation)

This step uses NASA GMAT to generate independent high-fidelity reference trajectories, enabling a non-circular comparison: PINN and SGP4 are each compared against GMAT, rather than comparing PINN against its own training data.

#### 7a. Install NASA GMAT

1. Download GMAT R2022a from: https://sourceforge.net/projects/gmat/files/GMAT/R2022a/
2. Install to the default location (e.g., `C:\GMAT\R2022a` on Windows)
3. Add the GMAT path to your `.env` file in the project root:

```
GMAT_PATH=C:\GMAT\R2022a
```

4. Verify the installation:

```bash
python gmat_config.py          # Check that GMAT is detected
python generate_gmat_data.py --validate   # Run a trivial GMAT script
```

#### 7b. Generate GMAT Trajectories

```bash
# Preview: generate GMAT scripts without running them
python generate_gmat_data.py --dry-run

# Test with a single satellite (ISS)
python generate_gmat_data.py --sat 25544

# Full catalog (all 20 satellites)
python generate_gmat_data.py
```

**GMAT force model per satellite:**
- 20x20 JGM-2 gravity harmonics
- MSISE-90 atmospheric drag model
- Solar radiation pressure (SRP)
- Sun and Moon third-body perturbations
- RungeKutta89 integrator, accuracy 1e-12

**What it produces:**
- `data/gmat_orbits/{norad_id}.npy` -- GMAT trajectories (5000 points, J2000 frame)
- `data/gmat_orbits/{norad_id}_meta.json` -- Metadata including force model details
- `data/gmat_scripts/{norad_id}.script` -- GMAT script files (human-readable)

**Runtime:** ~2--5 minutes per satellite (depends on GMAT installation)

#### 7c. Train PINN on GMAT Data

```bash
python train_real_orbits.py --data-source gmat
```

Same training pipeline as Step 6, but using GMAT trajectories instead of SGP4.

**What it produces:**
- `data/gmat_results/{norad_id}_results.json` -- Per-satellite results

#### 7d. Run the Hypothesis Test (PINN vs SGP4 vs GMAT)

```bash
python compare_pinn_vs_sgp4.py
```

For each satellite:
1. Loads GMAT ground truth
2. Trains PINN on first 20% of GMAT data (~1 orbit)
3. Propagates SGP4 from the same epoch (converted to J2000 frame)
4. Compares both against GMAT truth over the remaining 80%
5. Paired t-test and Wilcoxon signed-rank test across all 20 satellites

**What it produces:**
- `data/gmat_results/{norad_id}_comparison.json` -- Per-satellite comparison
- `data/gmat_results/hypothesis_test_summary.json` -- Aggregate statistics
- `figures/pinn_vs_sgp4_comparison.png` -- Bar chart
- `figures/pinn_vs_sgp4_scatter.png` -- Scatter plot
- `figures/pinn_improvement_histogram.png` -- Improvement distribution

**Runtime:** ~60 minutes for all 20 satellites

### Quick Reference: Complete Reproduction in One Block

```bash
# Clone and setup
git clone https://github.com/deusalter/ISEF-PINN.git
cd ISEF-PINN
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Synthetic experiments (Steps 1-5)
python generate_data.py
python train_baseline.py
python train_pinn.py
python train_pinn_j2.py
python evaluate.py

# Real satellite catalog (Step 6)
python download_tle.py
python generate_real_data.py
python train_real_orbits.py
python analyze_catalog.py

# GMAT ground truth (Step 7, requires GMAT installed)
python generate_gmat_data.py
python train_real_orbits.py --data-source gmat
python compare_pinn_vs_sgp4.py
```

### Expected Runtimes

| Step | Script | Approximate Time |
|---|---|---|
| 1. Synthetic data | `generate_data.py` | < 10 seconds |
| 2. Vanilla MLP | `train_baseline.py` | 1--3 minutes |
| 3. Two-body PINN | `train_pinn.py` | 10--15 minutes |
| 4. J2 PINN | `train_pinn_j2.py` | 15--20 minutes |
| 5. Evaluation | `evaluate.py` | < 60 seconds |
| 6a. TLE download | `download_tle.py` | < 10 seconds |
| 6b. SGP4 propagation | `generate_real_data.py` | < 30 seconds |
| 6c. Catalog training | `train_real_orbits.py` | 50--70 minutes |
| 6d. Analysis | `analyze_catalog.py` | < 30 seconds |
| 7b. GMAT data | `generate_gmat_data.py` | 2--5 min/satellite |
| 7c. GMAT training | `train_real_orbits.py --data-source gmat` | 50--70 minutes |
| 7d. Hypothesis test | `compare_pinn_vs_sgp4.py` | 50--70 minutes |

Runtimes measured on an Intel Core i7 laptop (no GPU). Apple M-series chips will be near the lower bound. Total wall time for full reproduction (Steps 1--7): approximately 4--5 hours.

---

## Key Results

### Two-Body Test Case (80/20 temporal train/test split, 3 orbital periods)

| Metric | Vanilla MLP | Fourier-PINN | Improvement |
|---|---|---|---|
| Test RMSE (km) | 10,199 | 145 | **98.6%** |
| Max position error (km) | 16,665 | 346 | **97.9%** |
| Hamiltonian energy drift (km^2/s^2) | 7.51 | 0.50 | **93.4%** |

### J2-Perturbed Test Case (20/80 temporal train/test split, 5 orbital periods)

| Metric | Vanilla MLP | Fourier-PINN | Improvement |
|---|---|---|---|
| Test RMSE (km) | 18,905 | 334 | **98.2%** |
| Max position error (km) | 25,746 | 518 | **97.9%** |
| Hamiltonian energy drift (km^2/s^2) | 12.80 | 1.23 | **90.4%** |

The J2 test case uses 20% training (~1 orbit), requiring the network to extrapolate 4 orbits. At t = 10,000 s the vanilla MLP diverges to ~10,000 km error while the PINN holds at 84 km (**99.1% improvement**).

### Real Satellite Catalog Validation (20 LEO satellites)

| Orbit Type | Satellites | Avg Vanilla RMSE (km) | Avg Best PINN (km) | Avg PINN J2 (km) |
|---|---|---|---|---|
| Low-inclination | CBERS 04A, TROPICS-01/02, ORBCOMM FM-5 | 20,511 | 3.05 | 3.87 |
| ISS-like | ISS, Tianhe CSS, Cygnus NG-19, Crew Dragon | 19,715 | 3.44 | 4.03 |
| Constellation | Starlink-1007/08/09/10 | 19,570 | 2.77 | 3.78 |
| Sun-synchronous | Landsat 9, Sentinel-6A, NOAA-20, Suomi NPP | 22,623 | 7.95 | 7.99 |
| Diverse | Hubble, GRACE-FO 1, Iridium 106, Cosmos 2251 debris | 22,415 | 8.27 | 10.09 |
| **All 20 satellites** | | **20,967** | **5.70** | **5.95** |

**Statistical significance:** The PINN outperforms the vanilla MLP on every single satellite without exception. 17/20 satellites achieve under 10 km test RMSE.

### Atmospheric Drag (Harris-Priester Model)

Over 5 orbital periods (~8 hours), atmospheric drag at LEO altitudes produces accelerations of order 10^-10 km/s^2 -- negligible compared to J2 (~10^-6 km/s^2). The J2+Drag model helps at low altitudes (CBERS at 628 km: 0.89 km, TIANHE at 390 km: 3.13 km) but can hurt at higher altitudes. This confirms J2 is the dominant perturbation for short-arc propagation.

### Detailed Per-Satellite Results (4 models each)

| NORAD | Name | Alt (km) | Inc (deg) | Vanilla (km) | Fourier NN (km) | PINN J2 (km) | PINN J2+Drag (km) | Best (km) |
|---|---|---|---|---|---|---|---|---|
| 44883 | CBERS 04A | 628 | 28.5 | 21,945 | 890 | 4.18 | **0.89** | 0.89 |
| 57320 | TROPICS-01 | 550 | 29.7 | 18,813 | 876 | **4.35** | 4.99 | 4.35 |
| 57321 | TROPICS-02 | 550 | 29.7 | 18,816 | 878 | **4.52** | 5.19 | 4.52 |
| 25063 | ORBCOMM FM-5 | 775 | 25.0 | 22,469 | 879 | **2.44** | 3.35 | 2.44 |
| 25544 | ISS (ZARYA) | 420 | 51.6 | 20,763 | 889 | 5.45 | **5.04** | 5.04 |
| 48274 | TIANHE (CSS) | 390 | 41.5 | 17,557 | 833 | 5.07 | **3.13** | 3.13 |
| 56227 | CYGNUS NG-19 | 415 | 51.6 | 20,690 | 890 | **3.01** | 6.00 | 3.01 |
| 58536 | CREW DRAGON | 420 | 51.6 | 19,849 | 888 | **2.57** | 3.52 | 2.57 |
| 44713 | STARLINK-1007 | 550 | 53.1 | 19,619 | 861 | 2.97 | **2.83** | 2.83 |
| 44714 | STARLINK-1008 | 550 | 53.0 | 19,541 | 861 | 5.18 | **2.63** | 2.63 |
| 44715 | STARLINK-1009 | 550 | 53.1 | 19,550 | 859 | 4.79 | **3.44** | 3.44 |
| 44716 | STARLINK-1010 | 550 | 53.0 | 19,570 | 858 | **2.17** | 4.28 | 2.17 |
| 49260 | LANDSAT 9 | 705 | 98.2 | 20,554 | 1,025 | **9.74** | 10.07 | 9.74 |
| 46984 | SENTINEL-6A | 830 | 66.0 | 21,096 | 923 | 3.22 | **3.03** | 3.03 |
| 43013 | NOAA-20 | 824 | 98.7 | 25,207 | 1,068 | **9.80** | 10.05 | 9.80 |
| 37849 | SUOMI NPP | 824 | 98.7 | 23,635 | 1,072 | **9.21** | 10.20 | 9.21 |
| 20580 | HUBBLE (HST) | 540 | 28.5 | 21,959 | 946 | **1.71** | 1.96 | 1.71 |
| 43476 | GRACE-FO 1 | 490 | 89.0 | 21,385 | 1,067 | 16.20 | **15.43** | 15.43 |
| 43070 | IRIDIUM 106 | 780 | 86.4 | 22,693 | 1,066 | 16.57 | **12.41** | 12.41 |
| 36508 | COSMOS 2251 DEB | 850 | 74.0 | 23,622 | 1,024 | **5.89** | 5.53 | 5.53 |

---

## Methodology

### Why GMAT as Ground Truth?

A common pitfall in ML-for-astrodynamics research is training a model on data from propagator X and then "validating" against propagator X -- this is circular and proves nothing about real-world accuracy. Our project addresses this:

| Stage | Data Source | Purpose |
|---|---|---|
| Training | GMAT (high-fidelity) | Teach the PINN orbital dynamics from the best available truth |
| PINN prediction | Neural network forward pass | Fast inference (~3 ms) |
| SGP4 prediction | SGP4 propagator | Industry-standard analytical propagator for comparison |
| Evaluation | GMAT (same high-fidelity truth) | Independent ground truth for both PINN and SGP4 |

GMAT's force model (20x20 gravity, MSISE-90 drag, SRP, Sun/Moon) is orders of magnitude more accurate than SGP4's simplified analytical model, making it a valid independent reference.

### Limitations and Caveats

1. **Information asymmetry.** The PINN trains on GMAT trajectories while SGP4 uses only a TLE-derived state — so the PINN has access to richer dynamics. This is deliberate: an operational PINN *would* be trained on the best available ephemeris, whereas SGP4 is constrained by its analytical theory. The comparison reflects realistic deployment, not a controlled ablation.

2. **What a fully symmetric test would require.** To eliminate the information advantage entirely, one would fit a same-complexity analytical model (e.g., Brouwer mean elements) on the identical GMAT training arc and compare extrapolation error head-to-head. This is left as future work.

3. **MC Dropout uncertainty is illustrative.** The PINN was trained without dropout, so applying dropout at inference provides only an approximate posterior. The resulting covariance ellipsoids and collision probabilities demonstrate the UQ *pipeline* but should not be treated as calibrated uncertainty estimates. A production system would retrain with dropout enabled and validate coverage empirically.

4. **Single architecture and seed.** All results use one Fourier-PINN architecture (8 frequencies, 64-wide, 3 hidden layers) with one random seed. Variance across seeds and sensitivity to hyperparameters have not been characterized.

### The Two-Body Orbital Mechanics Problem

A satellite in low Earth orbit obeys Newton's gravitational law:

```
d^2r/dt^2 = -MU * r / ||r||^3
```

where MU = 398,600.4418 km^3/s^2. This project uses a 400 km circular LEO as the reference orbit (r = 6,778.137 km, v_circ = 7.6686 km/s, T = 5,554 s). Ground-truth trajectories are produced by SciPy's DOP853 integrator with rtol = atol = 1e-12.

### Why Vanilla Neural Networks Fail at Extrapolation

A fully-connected MLP trained with MSE loss learns a smooth interpolant of the training data. With no knowledge of the governing ODE, its predictions degrade rapidly outside the training window -- accumulating position errors exceeding 10,000 km (larger than the orbital radius itself).

### Fourier Feature Encoding

The key architectural innovation is a Fourier feature input layer:

```
t_norm  ->  [sin(t), cos(t), sin(2t), cos(2t), ..., sin(Nt), cos(Nt)]
```

This provides sin/cos basis functions at integer harmonics of the orbital frequency, giving the network natural building blocks for periodic dynamics. Higher harmonics capture elliptical and perturbation effects. The number of harmonics is limited to avoid k^2 amplification in the second-derivative physics loss.

### Physics-Informed Training via Automatic Differentiation

The PINN adds a physics residual loss computed via PyTorch autograd:

```
pos_pred(t) = model(t)                          # forward pass
vel_pred(t) = d(pos_pred)/dt     via autograd   # exact first derivative
acc_pred(t) = d(vel_pred)/dt     via autograd   # exact second derivative

residual = acc_pred + MU * pos_pred / ||pos_pred||^3    (should be zero)
```

This residual is evaluated on collocation points spanning the **full** time domain (including the test region), so the physics constraint guides the network in regions with no labelled data.

### The Normalization Trick

Physical units (km, s) create large numerical imbalances. We normalize so that `MU * t_ref^2 / r_ref^3 = 1.0` exactly, making the physics residual O(1) and naturally balanced with the data MSE.

### J2 Perturbation

Earth's oblateness (J2 = 1.08263e-3) introduces secular orbital precession. The J2 acceleration is:

```
a_J2_x = -1.5 * J2 * MU * R_E^2 / r^5 * x * (1 - 5*(z/r)^2)
a_J2_y = -1.5 * J2 * MU * R_E^2 / r^5 * y * (1 - 5*(z/r)^2)
a_J2_z = -1.5 * J2 * MU * R_E^2 / r^5 * z * (3 - 5*(z/r)^2)
```

The physics loss also includes J3 and J4 harmonics for improved fidelity.

### Atmospheric Drag (Harris-Priester Model)

A torch-differentiable Harris-Priester density model provides drag acceleration:

```
a_drag = -0.5 * (Cd*A/m) * rho(h) * |v_rel| * v_rel
```

where `rho(h)` is altitude-dependent density and `v_rel` accounts for Earth's co-rotating atmosphere.

### Curriculum Training

Training uses a cosine-annealing learning rate (1e-3 to 1e-5) with gradual physics weight increase:

| Phase | Epochs | Physics weight | Purpose |
|---|---|---|---|
| 1. Data warmup | 1--2000 | 0.00 | Learn orbit shape from data only |
| 2. Gentle physics | 2001--5000 | 0.01 | Introduce physics without disrupting data fit |
| 3. Moderate physics | 5001--8000 | 0.05 | Strengthen physics constraints |
| 4. Full physics | 8001--10000 | 0.15 | Full composite loss |

At phase boundaries, the Adam optimizer is refreshed to reset stale momentum.

---

## Architecture

```
                VANILLA MLP                          FOURIER-PINN
          (data loss only)               (data loss + physics loss)

t_norm -----> [Linear 64] --tanh-->      t_norm ---> Fourier Encoding:
              [Linear 64] --tanh-->                  [sin(kt) for k=1..N_FREQ]
              [Linear 64] --tanh-->                  [cos(kt) for k=1..N_FREQ]
              [Linear  3] -------->                  = 2*N_FREQ input features
               (x, y, z)                                  |
                   |                          +-----------+-----------+
                   v                          |                       |
               MSE Loss                 [Linear 64] --tanh-->   sec_head (linear)
                                        [Linear 64] --tanh-->      * t
                                        [Linear 64] --tanh-->       |
                                        [Linear  3] ---------> + <--+
                                          periodic part      secular drift
                                                |
                                                |-----> MSE Loss (training data)
                                                |
                                     autograd   |   autograd
                                    d/dt_norm   |  d^2/dt_norm^2
                                                |
                                          vel_pred, acc_pred
                                                |
                                                v
                                     Physics Residual Loss
                                     acc + pos/||pos||^3 - a_J2 - a_J3 - a_J4 = 0
                                     (on 1000 collocation points
                                      spanning FULL time domain)
```

Both networks use `tanh` activations, Xavier-uniform initialization, `float64` precision, Adam optimizer with cosine-annealing LR, and gradient clipping (max_norm = 1.0). The secular drift head captures slow J2-induced precession (RAAN, argument of perigee) that a purely periodic model cannot represent.

---

## Repository Structure

```
ISEF-PINN/
|
|-- README.md                    # This file
|-- requirements.txt             # Python dependencies (pip install -r requirements.txt)
|-- .gitignore
|
|-- # --- Synthetic Experiments ---
|-- generate_data.py             # Integrate two-body and J2 ODEs; save ground-truth .npy files
|-- train_baseline.py            # Train vanilla MLP baseline (data loss only)
|-- train_pinn.py                # Train two-body PINN (Phase 3)
|-- train_pinn_j2.py             # Train J2-perturbed Fourier-PINN + matched vanilla baseline
|-- evaluate.py                  # Compute metrics and generate all publication figures
|-- plotting.py                  # Plotting helper library
|
|-- # --- Real Satellite Pipeline ---
|-- satellite_catalog.py         # Curated catalog of 20 LEO satellites (5 orbit types)
|-- download_tle.py              # Download TLEs from Space-Track (with offline fallbacks)
|-- generate_real_data.py        # SGP4 propagation of TLEs to dense trajectories
|-- train_real_orbits.py         # Train 4 models per satellite (--data-source sgp4|gmat)
|-- analyze_catalog.py           # Statistical analysis and publication figures
|
|-- # --- GMAT Integration ---
|-- gmat_config.py               # GMAT path detection and installation validation
|-- frame_conversion.py          # TEME <-> J2000 coordinate frame conversion (astropy)
|-- generate_gmat_data.py        # Generate GMAT scripts, run them, parse output (--dry-run, --validate)
|-- compare_pinn_vs_sgp4.py      # Hypothesis test: PINN vs SGP4 against GMAT ground truth
|
|-- # --- Additional Tools ---
|-- compare_sgp4.py              # SGP4 vs numerical integrator comparison
|-- conjunction_screening.py     # Monte Carlo conjunction assessment demo
|
|-- src/
|   |-- __init__.py
|   |-- models.py                # Canonical FourierPINN and VanillaMLP architectures
|   |-- physics.py               # Physical constants, ODEs, normalization, torch residuals
|   `-- atmosphere.py            # Harris-Priester density model + drag acceleration (torch)
|
|-- data/
|   |-- real_orbits/             # SGP4-propagated trajectories (.npy + _meta.json per satellite)
|   |-- real_results/            # Per-satellite training results + summary.json
|   |-- gmat_orbits/             # GMAT-propagated trajectories (generated by Step 7)
|   |-- gmat_results/            # GMAT-based comparison results (generated by Step 7)
|   |-- gmat_scripts/            # GMAT .script files (generated by Step 7)
|   `-- tle_catalog/             # Downloaded TLE files (.tle + .json per satellite)
|
|-- models/                      # Saved PyTorch model weights (.pt)
|-- figures/                     # All output plots (PNG) and tables (TXT, TEX)
`-- outputs/                     # Additional output plots
```

---

## Output Figures

### Synthetic Experiments

| File | Contents |
|---|---|
| `figures/3d_comparison.png` | 3D trajectory: ground truth, vanilla MLP, Fourier-PINN |
| `figures/2d_comparison.png` | X-Y plane projection with train/test boundary |
| `figures/energy_conservation.png` | Hamiltonian energy over time |
| `figures/loss_convergence.png` | Training loss curves |
| `figures/rmse_over_time.png` | Position error vs time |
| `figures/error_distribution.png` | Error histograms |
| `figures/j2_*.png` | Same set of figures for the J2-perturbed case |

### Real Satellite Catalog

| File | Contents |
|---|---|
| `figures/catalog_boxplot.png` | RMSE distributions across 20 satellites (4 models) |
| `figures/catalog_scatter.png` | PINN vs Vanilla RMSE scatter, color-coded by orbit type |
| `figures/catalog_correlations.png` | Improvement % vs altitude, inclination, eccentricity |
| `figures/catalog_summary.txt` | Full text summary table |
| `figures/catalog_table.tex` | LaTeX-ready results table |

### GMAT Comparison (generated by Step 7)

| File | Contents |
|---|---|
| `figures/pinn_vs_sgp4_comparison.png` | Bar chart: PINN vs SGP4 RMSE per satellite |
| `figures/pinn_vs_sgp4_scatter.png` | Scatter: PINN RMSE vs SGP4 RMSE |
| `figures/pinn_improvement_histogram.png` | Distribution of PINN improvement over SGP4 |

---

## Troubleshooting

### "No TLE available" for some satellites
The repo includes hardcoded fallback TLEs for all 20 satellites. If `download_tle.py` reports missing TLEs, the fallbacks will be used automatically. No Space-Track account is needed.

### GMAT not found
If you see `FileNotFoundError: GMAT installation not found`, either:
1. Install GMAT from https://sourceforge.net/projects/gmat/files/GMAT/R2022a/
2. Set `GMAT_PATH=C:\GMAT\R2022a` in a `.env` file in the project root
3. Skip Step 7 entirely -- the SGP4 pipeline (Steps 1--6) works independently

### CUDA / GPU issues
No GPU is needed. If you encounter CUDA errors, ensure PyTorch CPU-only is installed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Numerical differences from reported results
Small numerical differences (< 1%) are expected due to:
- Different CPU architectures (x86 vs ARM)
- PyTorch version differences
- Random seed behavior across platforms

The `torch.manual_seed(42)` call ensures reproducibility on the same hardware.

---

## Satellite Catalog

The 20 satellites span 5 orbit types, with altitudes 388--853 km and inclinations 25--99 degrees:

| Group | Satellites | Alt (km) | Inc (deg) |
|---|---|---|---|
| Low-inclination | CBERS 04A, TROPICS-01, TROPICS-02, ORBCOMM FM-5 | 548--782 | 25--30 |
| ISS-like | ISS, Tianhe CSS, Cygnus NG-19, Crew Dragon C212 | 388--420 | 42--52 |
| Constellation | Starlink-1007, -1008, -1009, -1010 | 547 | 53 |
| Sun-synchronous | Landsat 9, Sentinel-6A, NOAA-20, Suomi NPP | 703--827 | 66--99 |
| Diverse | Hubble, GRACE-FO 1, Iridium 106, Cosmos 2251 debris | 512--853 | 29--89 |

---

## Future Work

1. **GMAT-validated accuracy bounds:** With GMAT ground truth, establish error bounds on PINN propagation accuracy as a function of prediction horizon.
2. **Long-arc drag regime:** Extend propagation to hundreds of orbits (days to weeks) where atmospheric drag causes measurable orbital decay.
3. **Transfer learning across constellations:** Fine-tune a PINN trained on one satellite to new satellites with fewer epochs.
4. **Real-time conjunction screening:** Deploy the trained PINN as a fast surrogate propagator (~3 ms inference) for Monte Carlo conjunction assessment.

---

## Citation

This project was developed for the **Intel International Science and Engineering Fair (ISEF) 2026**, category: Systems Software / Computational and Systems Science.

```
@misc{namboori2026pinn,
  author = {Namboori, Abhinav},
  title  = {Physics-Informed Neural Networks for High-Fidelity Orbital
            Propagation and Real-Time LEO Conjunction Assessment},
  year   = {2026},
  note   = {ISEF 2026}
}
```

**License:** MIT
