# Physics-Informed Neural Networks for High-Fidelity Orbital Propagation and Real-Time LEO Conjunction Assessment

## Abstract

This project develops a Physics-Informed Neural Network (PINN) framework for propagating low-Earth-orbit (LEO) satellite trajectories by embedding gravitational dynamics directly into the neural network training loss via PyTorch automatic differentiation. A Fourier feature encoding layer provides the network with natural periodicity building blocks (sin/cos at orbital-frequency harmonics), enabling robust extrapolation of orbital dynamics. Unlike conventional data-driven networks that treat orbital mechanics as a black box, the PINN enforces the governing equations of motion at a dense set of collocation points spanning the full time domain -- including regions where labelled data are absent. On the two-body (Keplerian) test case, the Fourier-PINN achieves a **98.6% reduction in position RMSE** and a **93.4% reduction in Hamiltonian energy drift** relative to an identically sized vanilla MLP baseline. On the more challenging J2-perturbed test case (which includes secular orbital precession from Earth's oblateness), the Fourier-PINN achieves a **98.2% test RMSE reduction** when extrapolating 4 orbits beyond a single-orbit training window, demonstrating that physics-informed learning with periodic feature encoding can produce fast, reliable orbital predictions for real-time conjunction screening.

---

## Key Results

### Two-Body Test Case (80/20 temporal train/test split, 3 orbital periods)

| Metric | Vanilla MLP | Fourier-PINN | Improvement |
|---|---|---|---|
| Test RMSE (km) | 10,199 | 145 | **98.6%** |
| Max position error (km) | 16,665 | 346 | **97.9%** |
| Hamiltonian energy drift (km^2/s^2) | 7.51 | 0.50 | **93.4%** |

### RMSE at Time Checkpoints

| Time (s) | Vanilla RMSE (km) | PINN RMSE (km) | Improvement |
|---|---|---|---|
| 100 | 14.0 | 2.1 | **84.7%** |
| 1,000 | 7.5 | 1.2 | **83.6%** |
| 5,000 | 6.6 | 0.9 | **86.8%** |
| 10,000 (full) | 7.0 | 1.3 | **81.7%** |

The Fourier feature encoding gives the PINN natural building blocks for periodic orbital motion, enabling robust extrapolation far beyond the training window. Over the full test set the PINN cuts position error by **98.6%** -- from 10,199 km (larger than Earth's radius) down to 145 km. Energy conservation -- a quantity not explicitly penalised in the loss -- improves by **93.4%**, confirming that physics-informed learning implicitly preserves conserved quantities of the dynamical system.

### J2-Perturbed Test Case (20/80 temporal train/test split, 5 orbital periods)

| Metric | Vanilla MLP | Fourier-PINN | Improvement |
|---|---|---|---|
| Test RMSE (km) | 18,905 | 334 | **98.2%** |
| Max position error (km) | 25,746 | 518 | **97.9%** |
| Hamiltonian energy drift (km^2/s^2) | 12.80 | 1.23 | **90.4%** |

### J2 RMSE at Time Checkpoints

| Time (s) | Vanilla RMSE (km) | PINN RMSE (km) | Improvement |
|---|---|---|---|
| 100 | 22.8 | 100.3 | -339.4% |
| 1,000 | 12.8 | 59.3 | -362.6% |
| 5,000 | 10.5 | 34.0 | -222.7% |
| 10,000 | 9,774 | 83.6 | **99.1%** |

The J2 test case uses Earth's oblateness perturbation (J2 = 1.08263e-3), which introduces secular orbital precession -- a non-periodic effect that is harder to learn than pure Keplerian dynamics. The training set is limited to 20% (~1 orbit), requiring the network to extrapolate 4 orbits into the future. In the training region (t < 1,000 s) the vanilla MLP fits the data more tightly because it memorises the training points, while the PINN trades short-range interpolation accuracy for global physical consistency. This trade-off pays off dramatically at longer horizons: at t = 10,000 s the vanilla MLP has diverged to nearly 10,000 km error while the PINN holds at 84 km (**99.1% improvement**). Overall test RMSE drops from 18,905 km to 334 km (**98.2%**).

---

## Architecture

```
                    VANILLA MLP                          FOURIER-PINN
              (data loss only)               (data loss + physics loss)

    t_norm -----> [Linear 64] --tanh-->      t_norm ---> Fourier Encoding:
                  [Linear 64] --tanh-->                  [sin(t), cos(t), sin(2t),
                  [Linear 64] --tanh-->                   cos(2t), sin(3t), cos(3t),
                  [Linear  3] -------->                   sin(4t), cos(4t), t]
                   (x, y, z)                              = 9 input features
                       |                                       |
                       v                                 [Linear 64] --tanh-->
                   MSE Loss                              [Linear 64] --tanh-->
                                                         [Linear 64] --tanh-->
                                                         [Linear  3] -------->
                                                          (x, y, z)
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
                                                   acc + pos/||pos||^3 = 0
                                                   (on 1600 collocation pts
                                                    spanning FULL time domain)
```

Both networks use `tanh` activations, Xavier-uniform weight initialisation, `float64` precision, Adam optimiser with cosine-annealing learning rate schedule (1e-3 to 1e-5), and gradient clipping (`max_norm = 1.0`). The Fourier feature encoding provides the PINN with sin/cos basis functions at 4 harmonics of the orbital frequency (k = 1, 2, 3, 4), giving the network natural building blocks for periodic dynamics. Limiting to 4 frequencies avoids k^2 amplification in the second-derivative physics loss that occurs with higher harmonics.

For the J2 experiment, the raw `t` input is omitted from the Fourier encoding (purely periodic: 8 features), ensuring the network output is exactly periodic with period 2*pi. This is critical because the J2 test set extends 5x beyond the training window -- any non-periodic component would cause catastrophic extrapolation failure.

---

## Repository Structure

```
ISEF_PINN/
|
|-- generate_data.py          # Integrate two-body and J2 ODEs; save ground-truth .npy files
|-- train_baseline.py         # Train vanilla MLP (data loss only); save model and predictions
|-- run_pinn_quick.py         # Train Fourier-PINN for two-body case (best results)
|-- train_pinn.py             # Original two-body PINN (no Fourier features)
|-- train_pinn_j2.py          # Train J2-perturbed Fourier-PINN + matched vanilla baseline
|-- evaluate.py               # Load all artifacts; compute metrics; generate all figures
|-- plotting.py               # Publication-quality plotting library used by evaluate.py
|-- requirements.txt          # Python package dependencies
|-- README.md                 # This file
|
|-- src/
|   |-- __init__.py
|   `-- physics.py            # Physical constants, ODE right-hand sides, normalization, residuals
|
|-- data/                     # Generated automatically; .npy arrays produced by the pipeline
|   |-- orbital_data.npy              # Two-body ground truth  (2000, 7): [t, x, y, z, vx, vy, vz]
|   |-- orbital_data_j2.npy           # J2-perturbed ground truth  (5000, 7)
|   |-- vanilla_predictions.npy       # Vanilla MLP predictions  (2000, 4): [t, x, y, z] km
|   |-- pinn_predictions.npy          # Fourier-PINN predictions  (2000, 4): [t, x, y, z] km
|   |-- pinn_j2_predictions.npy       # J2 Fourier-PINN predictions  (5000, 4)
|   |-- vanilla_j2_predictions.npy    # J2 Vanilla MLP predictions  (5000, 4)
|   |-- vanilla_loss_history.npy      # Per-epoch MSE for vanilla MLP  (5000,)
|   `-- pinn_loss_history.npy         # Per-epoch [total, data, physics] for PINN  (10000, 3)
|
|-- models/                   # Saved PyTorch state dicts
|   |-- vanilla_mlp.pt
|   |-- pinn_twobody.pt
|   `-- pinn_j2.pt
|
`-- figures/                  # All output plots (PNG) and RMSE checkpoint table (TXT)
    |-- 3d_comparison.png            # Two-body: 3D trajectory comparison
    |-- 2d_comparison.png            # Two-body: X-Y plane projection
    |-- energy_conservation.png      # Two-body: Hamiltonian energy over time
    |-- loss_convergence.png         # Two-body: training loss curves
    |-- rmse_over_time.png           # Two-body: position error vs time
    |-- error_distribution.png       # Two-body: error histograms
    |-- rmse_table.txt               # Two-body: RMSE at time checkpoints
    |-- j2_3d_comparison.png         # J2: 3D trajectory comparison
    |-- j2_2d_comparison.png         # J2: X-Y plane projection
    |-- j2_energy_conservation.png   # J2: Hamiltonian energy over time
    |-- j2_rmse_over_time.png        # J2: position error vs time
    |-- j2_error_distribution.png    # J2: error histograms
    `-- j2_rmse_table.txt            # J2: RMSE at time checkpoints
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy matplotlib torch
```

No GPU is required. All models are trained and evaluated on CPU using `float64` precision.

---

## Quick Start: Reproducing Results

Run the scripts in order from the project root directory:

```bash
# Step 1 -- Generate ground-truth trajectory data (DOP853, rtol=atol=1e-12)
python3 generate_data.py

# Step 2 -- Train vanilla MLP baseline (5,000 epochs, ~2 min)
python3 train_baseline.py

# Step 3 -- Train two-body Fourier-PINN (10,000 epochs with curriculum, ~10-15 min)
python3 run_pinn_quick.py

# Step 4 -- Train J2-perturbed Fourier-PINN + matched vanilla baseline (~15-20 min)
python3 train_pinn_j2.py

# Step 5 -- Compute all metrics and generate publication figures
python3 evaluate.py
```

### Expected Runtimes (CPU, no GPU)

| Step | Script | Approximate Wall Time |
|---|---|---|
| Data generation | `generate_data.py` | < 10 seconds |
| Vanilla MLP training | `train_baseline.py` | 1--3 minutes (5,000 epochs) |
| Two-body Fourier-PINN | `run_pinn_quick.py` | 10--15 minutes (10,000 epochs) |
| J2 Fourier-PINN | `train_pinn_j2.py` | 15--20 minutes (10,000 epochs x 3 models) |
| Evaluation and plotting | `evaluate.py` | < 60 seconds |

Runtimes vary depending on hardware. A modern laptop CPU (Apple M-series or Intel Core i7/i9) will fall near the lower bound of each range.

---

## Methodology

### The Two-Body Orbital Mechanics Problem

A satellite in low Earth orbit obeys Newton's gravitational law. For a point mass subject only to Earth's monopole gravity, the position vector **r** = [x, y, z] evolves as:

```
d^2r/dt^2 = -MU * r / ||r||^3
```

where MU = 398,600.4418 km^3/s^2 is Earth's gravitational parameter. This project uses a 400 km circular LEO as the reference orbit (r = 6,778.137 km, v_circ = 7.6686 km/s, T = 5,554 s). Ground-truth trajectories are produced by SciPy's DOP853 integrator (8th-order Runge-Kutta) with tolerances rtol = atol = 1e-12, giving effectively machine-precision reference data.

### Why Vanilla Neural Networks Fail at Extrapolation

A fully-connected MLP trained to map time to position with a mean-squared-error loss learns a smooth interpolant of the training data. Because the network has no knowledge of the governing differential equation, its predictions degrade rapidly outside the training window. On the test set (the final 20% of the trajectory), the vanilla MLP accumulates position errors exceeding 10,000 km -- larger than the orbital radius itself.

### Fourier Feature Encoding

The key architectural innovation is a Fourier feature input layer that maps scalar time to a vector of sin/cos features at integer harmonics of the orbital frequency:

```
t_norm  ->  [sin(t), cos(t), sin(2t), cos(2t), sin(3t), cos(3t), sin(4t), cos(4t)]
```

In normalized time (where t_ref = T/(2*pi)), the orbital period corresponds to t_norm = 2*pi, so the k=1 Fourier feature directly captures the fundamental orbital frequency. Higher harmonics (k=2,3,4) capture elliptical and perturbation effects. The number of harmonics is limited to 4 to avoid k^2 amplification in the second-derivative physics loss: since d^2/dt^2 sin(kt) = -k^2 sin(kt), high-frequency features amplify the physics residual and fight the data fit.

For the two-body case (20% extrapolation), a raw `t` feature is appended to allow slight aperiodic corrections. For the J2 case (5x extrapolation), the encoding is **purely periodic** (no raw `t`), ensuring the network output is exactly periodic with period 2*pi. This prevents catastrophic extrapolation failure when the test set extends far beyond the training window.

### How PINNs Embed Physics via Automatic Differentiation

The Fourier-PINN adds a physics residual term to the training loss. PyTorch's autograd engine differentiates the network output with respect to the scalar input time, providing exact velocity and acceleration predictions without finite differences:

```
pos_pred(t)  = model(t)                           # forward pass
vel_pred(t)  = d(pos_pred) / dt    via autograd   # first derivative
acc_pred(t)  = d(vel_pred) / dt    via autograd   # second derivative
```

The two-body physics residual is then:

```
residual = acc_pred + MU * pos_pred / ||pos_pred||^3   (must equal zero)
```

This residual is minimised on a dense set of collocation points that span the entire time domain, including the test region where no labelled data exist. The physics constraint therefore acts as a global regulariser, guiding the network to produce physically consistent trajectories everywhere.

### The Normalization Trick: MU * t_ref^2 / r_ref^3 = 1

Directly using physical units (km, s) inside the physics loss introduces large numerical imbalances, because MU ~ 4 x 10^5 km^3/s^2 while network outputs are O(1). This project adopts a reference-value normalization:

```
r_norm = r / r_ref           (r_ref = 6,778.137 km)
t_norm = t / t_ref           (t_ref = T / 2*pi = 883.9 s)
v_norm = v / v_ref           (v_ref = r_ref / t_ref = 7.6686 km/s)
```

By construction, `MU * t_ref^2 / r_ref^3 = 1.0` exactly (this follows from the definition t_ref = sqrt(r_ref^3 / MU)). In normalized coordinates the two-body ODE reduces to:

```
d^2(r_norm)/d(t_norm)^2 + r_norm / ||r_norm||^3 = 0
```

No large physical constants appear, the residual is O(1), and the physics loss is naturally balanced with the data MSE loss (also O(1) in normalized coordinates).

### Composite Loss Function

```
L_total = L_data + lambda(epoch) * L_physics

L_data    = MSE( model(t_train), pos_train )
L_physics = mean( || acc_norm + pos_norm / ||pos_norm||^3 ||^2 )
            evaluated on 1600 collocation points spanning full time domain
```

The physics weight `lambda` follows a curriculum schedule (see below) that gradually increases from 0 to its final value, preventing the physics loss from overwhelming the data fit early in training.

### Curriculum Training

Training uses a cosine-annealing learning rate schedule (1e-3 to 1e-5) and proceeds in phases:

**Two-body (4-phase curriculum):**

| Phase | Epochs | lambda | Purpose |
|---|---|---|---|
| 1. Data warmup | 1--3000 | 0.0 | Learn orbit shape from data only |
| 2. Gentle physics | 3001--5000 | 0.01 | Introduce physics without disrupting data fit |
| 3. Moderate physics | 5001--8000 | 0.1 | Strengthen physics constraints |
| 4. Full physics | 8001--10000 | 1.0 | Full composite loss (physics residual already small) |

**J2-perturbed (2-phase with fresh optimizer):**

| Phase | Epochs | lambda | Purpose |
|---|---|---|---|
| 1. Data warmup | 1--3000 | 0.0 | Learn single-orbit shape from data only |
| 2. PINN phase | 3001--10000 | 0.05 | J2 physics constraint with moderate weight |

At the warmup-to-PINN transition, the optimizer is refreshed (new Adam instance at half the initial learning rate) to reset accumulated momentum, preventing stale gradient statistics from fighting the newly introduced physics loss.

This curriculum approach is critical: applying lambda = 1.0 from the start causes the second-derivative physics loss to dominate and destroy the data fit, yielding results **worse** than a vanilla MLP.

### J2 Perturbation Extension

`train_pinn_j2.py` extends the framework to include Earth's J2 oblateness perturbation:

```
a_J2_x = factor * x * (1 - 5 * (z/r)^2)
a_J2_y = factor * y * (1 - 5 * (z/r)^2)
a_J2_z = factor * z * (3 - 5 * (z/r)^2)
factor  = -1.5 * J2 * MU * R_Earth^2 / r^5      (J2 = 1.08263e-3)
```

In the J2 experiment the training split is reduced to 20% (one orbital period out of five), making the extrapolation challenge substantially harder: the network must predict 4 orbits into the future from a single orbit of training data. The Fourier encoding for J2 is **purely periodic** (no raw `t` input), ensuring the network output repeats with period 2*pi and preventing the tanh neurons from saturating when extrapolating 5x beyond the training window. The normalized J2 physics loss is derived analogously to the two-body case, with the dimensionless Earth radius `R_norm = R_Earth / r_ref` absorbing the physical constants.

---

## Output Figures

`evaluate.py` generates the following figures in the `figures/` directory. All figures are saved at 300 dpi and are suitable for ISEF poster board reproduction.

**Two-body figures:**

| File | Contents |
|---|---|
| `3d_comparison.png` | 3-D trajectory: ground truth (blue), vanilla NN (red), Fourier-PINN (green) |
| `2d_comparison.png` | Top-down X-Y projection with train/test boundary marker |
| `energy_conservation.png` | Specific orbital energy H(t) over time; flat = conservation, drift = physics violation |
| `loss_convergence.png` | Two-panel: total loss comparison (left); PINN data vs physics loss decomposition (right) |
| `rmse_over_time.png` | Per-timestep position error vs time in orbital periods |
| `error_distribution.png` | Side-by-side error histograms with mean and max annotated |
| `rmse_table.txt` | Plain-text RMSE at t = 100 s, 1,000 s, 5,000 s, and 10,000 s |

**J2-perturbed figures:**

| File | Contents |
|---|---|
| `j2_3d_comparison.png` | 3-D trajectory comparison for J2-perturbed orbit (5 orbits) |
| `j2_2d_comparison.png` | X-Y projection showing 1-orbit training vs 4-orbit extrapolation |
| `j2_energy_conservation.png` | Energy drift comparison under J2 perturbation |
| `j2_rmse_over_time.png` | Position error over time showing PINN dominance at longer horizons |
| `j2_error_distribution.png` | Error histograms for J2 test case |
| `j2_rmse_table.txt` | J2 RMSE at time checkpoints |

---

## Future Work

1. **J2 perturbation at scale:** Extend the J2-PINN to longer propagation horizons (tens of orbits) and inclined orbits where the out-of-plane J2 effect is stronger.

2. **Real TLE data:** Train and evaluate on real Two-Line Element (TLE) sets from the NORAD catalog, bridging the gap between idealised test cases and operational orbit determination.

3. **SGP4 comparison:** Benchmark the PINN against the industry-standard SGP4/SDP4 propagator used by NORAD for conjunction screening, comparing both accuracy and computational throughput.

4. **Drag and solar radiation pressure:** Incorporate non-conservative perturbations (atmospheric drag, solar radiation pressure) into the physics loss, which would require modelling time-varying atmospheric density.

5. **Multi-object conjunction screening:** Deploy the trained PINN as a fast surrogate propagator within a Monte Carlo conjunction assessment framework, leveraging the ~3-10 ms inference latency for real-time screening of large satellite catalogs.

6. **Adaptive collocation:** Implement adaptive collocation point placement that concentrates sampling where the physics residual is largest, improving training efficiency.

---

## Citation and License

This project was developed for submission to the **Intel International Science and Engineering Fair (ISEF) 2026** in the category of Systems Software / Computational and Systems Science.

```
@misc{namboori2026pinn,
  author = {Namboori, Abhinav},
  title  = {Physics-Informed Neural Networks for High-Fidelity Orbital
            Propagation and Real-Time LEO Conjunction Assessment},
  year   = {2026},
  note   = {ISEF 2026}
}
```

**License:** MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions: the above copyright notice and this permission notice shall be included in all copies or substantial portions of the software. The software is provided "as is", without warranty of any kind.
