#!/usr/bin/env python3
"""
Statistical analysis and figure generation for the ISEF PINN
real satellite catalog experiment.

Reads per-satellite results from data/real_results/{norad_id}_results.json,
runs paired t-tests, and produces publication-quality figures.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics import R_EARTH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "real_results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Orbit-type visual style mapping
# ---------------------------------------------------------------------------
ORBIT_COLORS = {
    "low-inclination": "blue",
    "ISS-like": "orange",
    "constellation": "purple",
    "sun-synchronous": "red",
    "diverse": "green",
}
ORBIT_MARKERS = {
    "low-inclination": "o",
    "ISS-like": "s",
    "constellation": "D",
    "sun-synchronous": "^",
    "diverse": "v",
}


# ===================================================================
# 1. Data Loading
# ===================================================================
def load_results():
    """Load all per-satellite JSON result files.

    Returns
    -------
    list[dict]
        One dictionary per satellite, sorted by NORAD ID.
    """
    pattern = os.path.join(RESULTS_DIR, "*_results.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No result files found in {RESULTS_DIR}")
        sys.exit(1)

    results = []
    for fpath in files:
        with open(fpath, "r") as f:
            results.append(json.load(f))

    results.sort(key=lambda r: r["norad_id"])
    print(f"Loaded {len(results)} satellite results from {RESULTS_DIR}")
    return results


# ===================================================================
# 2. Statistical Tests
# ===================================================================
def run_statistics(results):
    """Run paired t-tests and print results to the console."""
    pinn = np.array([r["fourier_pinn_test_rmse_km"] for r in results])
    fnn = np.array([r["fourier_nn_test_rmse_km"] for r in results])
    vanilla = np.array([r["vanilla_test_rmse_km"] for r in results])
    n = len(results)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS  (paired t-tests, two-sided)")
    print("=" * 70)

    for label, baseline in [("Vanilla", vanilla), ("Fourier NN", fnn)]:
        diff = baseline - pinn
        t_stat, p_val = stats.ttest_rel(baseline, pinn)
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci95 = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se_diff)

        print(f"\n--- PINN vs {label} ---")
        print(f"  N             = {n}")
        print(f"  Mean diff     = {mean_diff:,.2f} km  ({label} - PINN)")
        print(f"  t-statistic   = {t_stat:.4f}")
        print(f"  p-value       = {p_val:.6g}")
        print(f"  95% CI        = [{ci95[0]:,.2f}, {ci95[1]:,.2f}] km")
        if p_val < 0.001:
            print("  Significance  : p < 0.001 ***")
        elif p_val < 0.01:
            print("  Significance  : p < 0.01  **")
        elif p_val < 0.05:
            print("  Significance  : p < 0.05  *")
        else:
            print("  Significance  : NOT significant (p >= 0.05)")

    print()


# ===================================================================
# 3. Box Plot
# ===================================================================
def plot_boxplot(results):
    """Box plot comparing test RMSE across three model families."""
    vanilla = [r["vanilla_test_rmse_km"] for r in results]
    fnn = [r["fourier_nn_test_rmse_km"] for r in results]
    pinn = [r["fourier_pinn_test_rmse_km"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))

    data = [vanilla, fnn, pinn]
    labels = ["Vanilla NN", "Fourier NN", "Fourier PINN"]
    colors = ["#e74c3c", "#c74cdb", "#27ae60"]  # red, magenta, green

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.45,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Overlay individual data points with small jitter
    rng = np.random.default_rng(42)
    for idx, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            s=30,
            zorder=5,
            alpha=0.85,
        )

    ax.set_yscale("log")
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Test RMSE (km)", fontsize=12)
    ax.set_title("Test RMSE Distribution Across LEO Satellite Catalog", fontsize=13)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.tick_params(axis="y", which="both", labelsize=10)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "catalog_boxplot.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved box plot -> {out}")


# ===================================================================
# 4. Scatter Plot  (Vanilla RMSE vs PINN RMSE, log-log)
# ===================================================================
def plot_scatter(results):
    """Scatter plot: Vanilla vs PINN test RMSE, coloured by orbit type."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Collect unique orbit types present in data
    orbit_types_present = sorted(set(r.get("orbit_type", "diverse") for r in results))

    for otype in orbit_types_present:
        subset = [r for r in results if r.get("orbit_type", "diverse") == otype]
        xs = [r["vanilla_test_rmse_km"] for r in subset]
        ys = [r["fourier_pinn_test_rmse_km"] for r in subset]
        ax.scatter(
            xs,
            ys,
            c=ORBIT_COLORS.get(otype, "gray"),
            marker=ORBIT_MARKERS.get(otype, "o"),
            label=otype.replace("_", " ").title(),
            s=55,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )

    # Diagonal y=x reference line
    all_vals = (
        [r["vanilla_test_rmse_km"] for r in results]
        + [r["fourier_pinn_test_rmse_km"] for r in results]
    )
    lo = min(all_vals) * 0.5
    hi = max(all_vals) * 2.0
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, zorder=1, label="y = x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Vanilla NN Test RMSE (km)", fontsize=12)
    ax.set_ylabel("Fourier PINN Test RMSE (km)", fontsize=12)
    ax.set_title("PINN vs Vanilla Test RMSE by Orbit Type", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "catalog_scatter.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved scatter plot -> {out}")


# ===================================================================
# 5. Correlations with Orbital Parameters
# ===================================================================
def plot_correlations(results):
    """1x3 subplot: improvement % vs altitude, inclination, eccentricity."""
    improvement = np.array([r["pinn_improvement_pct"] for r in results])
    altitude = np.array([r["a_km"] - R_EARTH for r in results])  # semi-major axis -> alt
    inclination = np.array([r["inc_deg"] for r in results])
    eccentricity = np.array([r["ecc"] for r in results])

    param_data = [
        ("Altitude (km)", altitude),
        ("Inclination (deg)", inclination),
        ("Eccentricity", eccentricity),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (xlabel, xvals) in zip(axes, param_data):
        ax.scatter(xvals, improvement, c="#2980b9", s=40, edgecolors="black", linewidths=0.4)

        # Linear trend line
        if len(xvals) > 1 and np.std(xvals) > 0:
            coeffs = np.polyfit(xvals, improvement, 1)
            xline = np.linspace(xvals.min(), xvals.max(), 100)
            ax.plot(xline, np.polyval(coeffs, xline), "r--", linewidth=1.2)

            r_val, p_val = stats.pearsonr(xvals, improvement)
            ax.text(
                0.05,
                0.05,
                f"r = {r_val:.3f}\np = {p_val:.3g}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            print(f"  Pearson r({xlabel:>18s} vs improvement) = {r_val:+.4f}  (p={p_val:.4g})")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("PINN Improvement (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle("PINN Improvement vs Orbital Parameters", fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "catalog_correlations.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlations plot -> {out}")


# ===================================================================
# 6. Summary Table (console + text file)
# ===================================================================
def print_summary_table(results):
    """Print a formatted summary table and save to figures/catalog_summary.txt."""
    header = (
        f"{'NORAD':>7s}  {'Name':<22s}  {'Alt(km)':>8s}  {'Inc(deg)':>8s}  "
        f"{'Vanilla':>10s}  {'FNN':>10s}  {'PINN':>10s}  {'Improv%':>8s}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]

    vanilla_all, fnn_all, pinn_all, improv_all = [], [], [], []

    for r in results:
        alt = r["a_km"] - R_EARTH
        line = (
            f"{r['norad_id']:>7d}  {r['name']:<22s}  {alt:>8.1f}  {r['inc_deg']:>8.2f}  "
            f"{r['vanilla_test_rmse_km']:>10.1f}  {r['fourier_nn_test_rmse_km']:>10.1f}  "
            f"{r['fourier_pinn_test_rmse_km']:>10.1f}  {r['pinn_improvement_pct']:>7.1f}%"
        )
        lines.append(line)
        vanilla_all.append(r["vanilla_test_rmse_km"])
        fnn_all.append(r["fourier_nn_test_rmse_km"])
        pinn_all.append(r["fourier_pinn_test_rmse_km"])
        improv_all.append(r["pinn_improvement_pct"])

    lines.append(sep)

    # Mean row
    mean_line = (
        f"{'':>7s}  {'MEAN':<22s}  {'':>8s}  {'':>8s}  "
        f"{np.mean(vanilla_all):>10.1f}  {np.mean(fnn_all):>10.1f}  "
        f"{np.mean(pinn_all):>10.1f}  {np.mean(improv_all):>7.1f}%"
    )
    lines.append(mean_line)

    # Std row
    std_line = (
        f"{'':>7s}  {'STD':<22s}  {'':>8s}  {'':>8s}  "
        f"{np.std(vanilla_all, ddof=1):>10.1f}  {np.std(fnn_all, ddof=1):>10.1f}  "
        f"{np.std(pinn_all, ddof=1):>10.1f}  {np.std(improv_all, ddof=1):>7.1f}%"
    )
    lines.append(std_line)
    lines.append(sep)

    table_str = "\n".join(lines)
    print("\n" + table_str)

    out = os.path.join(FIGURES_DIR, "catalog_summary.txt")
    with open(out, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved summary table -> {out}")


# ===================================================================
# 7. LaTeX Table
# ===================================================================
def save_latex_table(results):
    """Write a LaTeX tabular environment to figures/catalog_table.tex."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Satellite catalog experiment: test RMSE (km) by model.}")
    lines.append(r"\label{tab:catalog}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"NORAD & Name & Alt (km) & Inc ($^\circ$) & Vanilla & Fourier NN & Fourier PINN & Improv. (\%) \\"
    )
    lines.append(r"\midrule")

    vanilla_all, fnn_all, pinn_all, improv_all = [], [], [], []

    for r in results:
        alt = r["a_km"] - R_EARTH
        # Escape any special LaTeX characters in the name
        name = r["name"].replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")
        row = (
            f"  {r['norad_id']} & {name} & {alt:.0f} & {r['inc_deg']:.1f} & "
            f"{r['vanilla_test_rmse_km']:.0f} & {r['fourier_nn_test_rmse_km']:.0f} & "
            f"{r['fourier_pinn_test_rmse_km']:.0f} & {r['pinn_improvement_pct']:.1f} \\\\"
        )
        lines.append(row)
        vanilla_all.append(r["vanilla_test_rmse_km"])
        fnn_all.append(r["fourier_nn_test_rmse_km"])
        pinn_all.append(r["fourier_pinn_test_rmse_km"])
        improv_all.append(r["pinn_improvement_pct"])

    lines.append(r"\midrule")
    lines.append(
        f"  & \\textbf{{Mean}} & & & "
        f"{np.mean(vanilla_all):.0f} & {np.mean(fnn_all):.0f} & "
        f"{np.mean(pinn_all):.0f} & {np.mean(improv_all):.1f} \\\\"
    )
    lines.append(
        f"  & \\textbf{{Std}} & & & "
        f"{np.std(vanilla_all, ddof=1):.0f} & {np.std(fnn_all, ddof=1):.0f} & "
        f"{np.std(pinn_all, ddof=1):.0f} & {np.std(improv_all, ddof=1):.1f} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = os.path.join(FIGURES_DIR, "catalog_table.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table -> {out}")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("  ISEF PINN -- Satellite Catalog Statistical Analysis")
    print("=" * 70)

    results = load_results()

    run_statistics(results)

    print("\nGenerating figures...")
    plot_boxplot(results)
    plot_scatter(results)

    print("\nCorrelation analysis:")
    plot_correlations(results)

    print_summary_table(results)
    save_latex_table(results)

    print("\nDone. All outputs saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
