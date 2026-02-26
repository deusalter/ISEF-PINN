# ISEF-PINN Project Context

## Hypothesis

If a Physics-Informed Neural Network (PINN) is used for orbital propagation, where the loss
function is constrained by the ordinary differential equations of orbital dynamics, then the
model will demonstrate a statistically significant reduction in Euclidean position error
compared to the SGP4 baseline.

## Training Strategy (CRITICAL)

We train on **GMAT** (NASA's General Mission Analysis Tool), NOT on SGP4. GMAT is our
high-fidelity ground truth (20x20 gravity harmonics, MSISE-90 atmosphere, solar radiation
pressure, third-body perturbations, RK89 integrator at 1e-12 accuracy). SGP4 is the
**baseline/control** that we are trying to beat. The PINN learns the physics from GMAT data
and aims to approximate GMAT-quality predictions at near-SGP4 inference speed.

## Target Metrics (7-Day LEO Simulation)

| Metric | SGP4 (Baseline) | PINN (Target) | GMAT (Truth) |
|---|---|---|---|
| Position RMSE (km) | ~12.4 km | < 2.0 km | 0.0 km |
| Max Position Error (km) | ~18.1 km | < 5.0 km | 0.0 km |
| Inference Time (ms) | ~15 ms | < 50 ms | ~4500 ms |
| Max Energy Drift (km^2/s^2) | ~10^-3 | ~10^-6 | ~10^-12 |

## Experimental Groups

- **Control**: SGP4 (industry-standard analytical propagator, no AI)
- **Experimental Group A**: Vanilla Neural Network (data-driven only, no physics)
- **Experimental Group B**: Physics-Informed Neural Network (AI + orbital mechanics)

## Key Architecture

- Fourier PINN with secular drift head
- Physics loss: J2 + J3 + J4 zonal harmonics + Harris-Priester atmospheric drag
- Curriculum training: warmup -> physics ramp-up -> fine-tuning
- Adaptive collocation points spanning train and test regions

## Project Structure

- `generate_data.py` / `generate_gmat_data.py` - Ground truth generation
- `train_baseline.py` - Vanilla MLP baseline (Phase 2)
- `train_pinn.py` - Two-body PINN (Phase 3)
- `train_pinn_j2.py` - J2 Fourier PINN (Phase 4)
- `train_real_orbits.py` - Main training pipeline for real satellites
- `compare_pinn_vs_sgp4.py` - GMAT-based hypothesis test (PINN vs SGP4)
- `evaluate.py` / `plotting.py` - Metrics and visualization
- `conjunction_screening.py` - Conjunction assessment demo
- `src/models.py` - Canonical model definitions (FourierPINN, VanillaMLP)
- `src/physics.py` - Physical constants and equations
- `src/atmosphere.py` - Harris-Priester atmosphere model
