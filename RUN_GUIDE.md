# ISEF-PINN Run Guide

## What This Project Does

Physics-Informed Neural Networks (PINNs) for satellite orbit propagation, tested against the industry-standard SGP4 propagator using NASA GMAT as independent ground truth. The core hypothesis test:

- **H0**: PINN and SGP4 have equal accuracy propagating LEO orbits
- **H1**: PINN achieves lower RMSE than SGP4 over the prediction window

20 LEO satellites across 5 orbit types (low-inclination, ISS-like, constellation, sun-synchronous, diverse).

---

## Prerequisites

### Software
- **Python 3.10+** with these packages:
  ```
  pip install numpy scipy matplotlib torch sgp4 spacetrack python-dotenv astropy
  ```
- **NASA GMAT R2022a** (optional but needed for generating ground truth data)
  - Install to default path or set `GMAT_PATH` environment variable
  - The pre-generated GMAT data in `data/gmat_orbits/` means you can skip GMAT installation if data already exists

### Hardware
- No GPU required. All training uses float64 on CPU by default.
- GPU (CUDA) is supported and used automatically if available — speeds up Neural ODE training significantly.
- Expect ~15-20 min per satellite for 7-day Neural ODE training on GPU, ~35 min on CPU.

### Space-Track Credentials (Optional)
For downloading real TLEs, create a `.env` file in the project root:
```
SPACETRACK_USER=your_email
SPACETRACK_PASS=your_password
```
Hardcoded fallback TLEs are included for all 20 satellites, so this is optional.

---

## Project Structure

```
ISEF-PINN/
├── src/
│   ├── models.py          # Neural network architectures (FourierPINN, NeuralODE)
│   ├── physics.py         # Physical constants, J2-J5 gravity, normalization
│   └── atmosphere.py      # Harris-Priester drag model (differentiable)
├── data/
│   ├── tle_catalog/       # TLE files for 20 satellites
│   ├── gmat_orbits/       # GMAT ground truth trajectories (.npy)
│   ├── gmat_results/      # PINN vs SGP4 comparison results (.json)
│   ├── gmat_scripts/      # Generated GMAT simulation scripts
│   ├── real_orbits/       # SGP4-propagated trajectories
│   └── real_results/      # Training results on SGP4 data
├── figures/               # Generated plots
├── outputs/               # Log files
└── [scripts described below]
```

---

## Two Modes of Operation

### 5-Orbit Mode (8 hours) — `FourierPINN`
- Architecture: Fourier-encoded PINN with secular drift head
- Maps time directly to position (function approximator)
- Fast to train (~3 min/satellite) but weaker at extrapolation
- 20% train / 80% test split

### 7-Day Mode (168 hours) — `NeuralODE` (recommended)
- Architecture: Neural ODE with RK4 integration + multiple shooting
- Integrates real orbital mechanics (J2-J5 gravity + atmospheric drag)
- NN learns only a small correction term (1,380 parameters)
- Much more accurate — this is the main result for the paper
- 20% train (33.6h) / 80% test (134.4h)
- Settings: 2000 epochs, dt=60s, 20 shooting segments

Both are PINNs — the physics is embedded in the loss function (5-orbit) or the ODE structure (7-day).

---

## Step-by-Step Pipeline

### Phase 1: Synthetic Data Validation (optional, ~30 min)

These steps validate the PINN approach on synthetic orbits before real satellites.

```bash
# Generate synthetic orbital data (two-body + J2-perturbed)
python generate_data.py

# Train vanilla MLP baseline (shows why plain NNs fail)
python train_baseline.py

# Train PINN on two-body problem
python train_pinn.py

# Train J2-perturbed Fourier PINN with extrapolation test
python train_pinn_j2.py

# Evaluate all synthetic results and generate figures
python evaluate.py
```

### Phase 2: Download TLEs

```bash
python download_tle.py
```
Downloads TLEs from Space-Track (or uses hardcoded fallbacks). Output: `data/tle_catalog/`.

### Phase 3: Generate GMAT Ground Truth

Skip this if `data/gmat_orbits/` already has `.npy` files.

```bash
# 5-orbit data (8 hours per satellite)
python generate_gmat_data.py

# 7-day data (168 hours per satellite)
python generate_gmat_data.py --long-arc

# Dry run (generate scripts only, don't run GMAT)
python generate_gmat_data.py --long-arc --dry-run

# Single satellite
python generate_gmat_data.py --long-arc --sat 25544
```

GMAT force model: JGM-2 20x20 gravity, MSISE-90 drag, SRP, Sun/Moon, RungeKutta89 at 1e-12 tolerance.

### Phase 4: Run the Hypothesis Test

This is the core experiment.

```bash
# 5-orbit comparison (FourierPINN, ~1 hour total)
python compare_pinn_vs_sgp4.py

# 7-day comparison (NeuralODE, ~5-7 hours total)
python compare_pinn_vs_sgp4.py --long-arc

# Single satellite (useful for testing)
python compare_pinn_vs_sgp4.py --long-arc --sat 25544
```

**What it does for each satellite:**
1. Loads GMAT ground truth from `data/gmat_orbits/`
2. Trains PINN on first 20% of data
3. Propagates SGP4 from TLE epoch, converts TEME → J2000
4. Compares both against GMAT truth over remaining 80%
5. Saves per-satellite results to `data/gmat_results/{norad_id}_comparison[_7day].json`
6. After all satellites: runs paired t-test, generates figures, saves summary

**Output files:**
- `data/gmat_results/{norad_id}_comparison_7day.json` — per-satellite metrics
- `data/gmat_results/hypothesis_test_summary.json` — aggregate statistics + t-test
- `figures/pinn_vs_sgp4_comparison.png` — bar chart
- `figures/pinn_vs_sgp4_scatter.png` — scatter plot
- `figures/pinn_improvement_histogram.png` — improvement distribution

### Phase 5: Recompute SGP4 Only (optional)

If you update TLEs and want to recompute SGP4 metrics without retraining PINN:

```bash
python recompute_sgp4.py            # 5-orbit
python recompute_sgp4.py --long-arc # 7-day
```

---

## Running the Full Pipeline at Once

```bash
bash run_full_pipeline.sh
```

This runs Phase 2 → Phase 3 → Phase 4 sequentially with logging to `outputs/`.

---

## Key Results to Look For

After the 7-day comparison completes, check `data/gmat_results/hypothesis_test_summary.json`:

```json
{
  "n_satellites": 20,
  "pinn_mean_rmse_km": ...,
  "sgp4_mean_rmse_km": ...,
  "t_statistic": ...,
  "p_value_one_sided": ...,
  "cohens_d": ...,
  "significant_at_005": true/false,
  "pinn_wins": ...
}
```

- **p_value_one_sided < 0.05** → statistically significant result
- **pinn_wins** → number of satellites where PINN beat SGP4
- **cohens_d** → effect size (negative = PINN better)

---

## Current Status

### 7-Day Results (as of last run)
- **11/20 satellites completed** with valid results — PINN wins all 11
- **9 satellites need retraining** (8 had corrupted results from interrupted runs, 1 was never run)
- Satellites needing rerun: CBERS 04A (44883), TROPICS-01 (57320), CREW DRAGON C212 (58536), SENTINEL-6A (46984), STARLINK-1007/1008/1009/1010 (44713-44716), COSMOS 2251 DEB (36508)

To rerun just the missing satellites:
```bash
python compare_pinn_vs_sgp4.py --long-arc --sat 44883
python compare_pinn_vs_sgp4.py --long-arc --sat 57320
python compare_pinn_vs_sgp4.py --long-arc --sat 58536
python compare_pinn_vs_sgp4.py --long-arc --sat 46984
python compare_pinn_vs_sgp4.py --long-arc --sat 44713
python compare_pinn_vs_sgp4.py --long-arc --sat 44714
python compare_pinn_vs_sgp4.py --long-arc --sat 44715
python compare_pinn_vs_sgp4.py --long-arc --sat 44716
python compare_pinn_vs_sgp4.py --long-arc --sat 36508
```

Each takes ~15-20 min on GPU, ~35 min on CPU. After all finish, rerun the full comparison to regenerate the summary:
```bash
python compare_pinn_vs_sgp4.py --long-arc
```

Or just recompute SGP4 metrics on existing results:
```bash
python recompute_sgp4.py --long-arc
```

---

## Important Notes

- **Do NOT change training hyperparameters** (NODE_EPOCHS=2000, NODE_DT=60, NODE_N_SEGMENTS=20). These were tuned for accuracy. Reducing epochs or increasing dt destroys results.
- **Seed is fixed** (seed=42) for reproducibility. Identical runs produce identical PINN results.
- **TLEs are synthetic** — they match the GMAT initial conditions exactly. This is correct because GMAT was initialized from those TLE elements. Using "real" TLEs from Space-Track would be an unfair comparison (different initial conditions).
- **Frame conversion**: SGP4 outputs TEME coordinates, GMAT outputs J2000. The `frame_conversion.py` handles TEME → J2000 conversion via astropy.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GMAT not found | Install GMAT R2022a or set `GMAT_PATH` env var. Skip if data already exists. |
| TLE download fails | Hardcoded fallbacks are used automatically. Check `.env` credentials if needed. |
| Training is slow | Use GPU if available. Don't reduce epochs below 2000. |
| Bad PINN RMSE (>100 km) | Delete the bad `_comparison_7day.json` and rerun that satellite. |
| SGP4 errors | Check TLE file exists in `data/tle_catalog/`. Rerun `download_tle.py`. |
| Out of memory | Reduce batch size or use CPU (set `DEVICE = "cpu"` in compare script). |
