"""
Run the article's decentralised EE algorithm and evaluate.
============================================================
Usage:
    python article_algorithm/run.py                  # 200 trials, default
    python article_algorithm/run.py --trials 500     # more trials
    python article_algorithm/run.py --verbose         # per-iteration output
    python article_algorithm/run.py --mode synchronous  # Jacobi updates

Outputs:
    results/article_algorithm_results.json
    results/figures/article_algorithm_*.png
"""

import sys
import os
import time
import json
import argparse

import numpy as np

# ---- path setup ----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import yaml
from core.network import MassiveMIMOChannel
from article_algorithm.algorithm import DecentralizedEEAlgorithm


# ======================================================================
# Configuration helpers
# ======================================================================

def load_config():
    """Load network configuration and derive algorithm parameters."""
    with open("config/network.yaml") as f:
        cfg = yaml.safe_load(f)["network"]

    p_max     = 10 ** (cfg["power"]["max_power_dBm"]     / 10) * 1e-3
    p_min     = 10 ** (cfg["power"]["min_power_dBm"]     / 10) * 1e-3
    p_circuit = 10 ** (cfg["power"]["circuit_power_dBm"] / 10) * 1e-3

    eps_sq    = (cfg["channel"]["error_magnitude_bs"] ** 2
                 + cfg["channel"]["error_magnitude_ue"] ** 2)
    gamma_max = 1.0 / eps_sq
    max_rate  = np.log2(1.0 + gamma_max)

    qos_frac  = cfg["qos"]["target_rate_fraction"] \
                if "target_rate_fraction" in cfg["qos"] \
                else cfg["qos"]["min_rate_percentage"] / 100.0
    qos_target_rate = qos_frac * max_rate

    return {
        "p_max":           p_max,
        "p_min":           p_min,
        "p_circuit":       p_circuit,
        "gamma_max":       gamma_max,
        "max_rate":        max_rate,
        "qos_target_rate": qos_target_rate,
        "gamma_min":       2.0 ** qos_target_rate - 1.0,
    }


# ======================================================================
# Main evaluation loop
# ======================================================================

def run_evaluation(n_trials: int = 200,
                   update_mode: str = "sequential",
                   verbose_trials: bool = False):
    """
    Evaluate the article algorithm over many channel realisations.

    Returns a results dict (also saved to JSON).
    """
    channel = MassiveMIMOChannel("config/network.yaml")
    cfg     = load_config()

    # ---- header ----
    print("=" * 70)
    print("ARTICLE ALGORITHM — Decentralised EE Maximisation (Zappone et al.)")
    print("=" * 70)
    print(f"\nNetwork : K = {channel.n_users} users,  M = {channel.n_antennas} antennas,  "
          f"B = {channel.bandwidth / 1e6:.0f} MHz")
    print(f"Power   : p_max = {cfg['p_max']*1e3:.1f} mW,  "
          f"p_circuit = {cfg['p_circuit']*1e3:.1f} mW")
    print(f"QoS     : θ_k = {cfg['qos_target_rate']:.3f} bit/s/Hz,  "
          f"γ̲ = {cfg['gamma_min']:.3f}")
    print(f"HW      : γ̄ = {cfg['gamma_max']:.0f}  "
          f"({10*np.log10(cfg['gamma_max']):.1f} dB)")
    print(f"Mode    : {update_mode}")
    print(f"\nRunning {n_trials} channel realisations …\n")

    # ---- storage ----
    all_ee         = []       # mean EE per trial  (bit/J)
    all_ee_per_user = []
    all_powers     = []       # mean power per trial (W)
    all_sinrs      = []       # mean SINR  (linear)
    all_rates      = []       # mean rate  (bit/s/Hz)
    all_iters      = []
    all_converged  = []
    all_qos_met    = []

    t0 = time.time()

    for trial in range(n_trials):
        channel.reset()

        algo = DecentralizedEEAlgorithm(
            channel=channel,
            p_max=cfg["p_max"],
            p_circuit=cfg["p_circuit"],
            qos_target_rate=cfg["qos_target_rate"],
            update_mode=update_mode,
        )

        powers, ees, hist = algo.solve(verbose=verbose_trials)

        sinrs = channel.compute_all_sinr(powers)
        rates = np.array([channel.compute_rate(s) for s in sinrs])
        qos_ok = bool(np.all(rates >= cfg["qos_target_rate"] * 0.95))

        all_ee.append(float(np.mean(ees)))
        all_ee_per_user.append(ees.tolist())
        all_powers.append(float(np.mean(powers)))
        all_sinrs.append(float(np.mean(sinrs)))
        all_rates.append(float(np.mean(rates)))
        all_iters.append(hist["n_outer_iter"])
        all_converged.append(hist["converged"])
        all_qos_met.append(qos_ok)

        if (trial + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{trial+1:4d}/{n_trials}]  "
                  f"elapsed {elapsed:.1f}s   "
                  f"avg EE = {np.mean(all_ee)/1e6:.1f} Mbit/J")

    elapsed = time.time() - t0

    # ---- statistics ----
    ee_M   = np.array(all_ee) / 1e6
    pw_mW  = np.array(all_powers) * 1e3
    sinr_a = np.array(all_sinrs)
    rate_a = np.array(all_rates)

    print(f"\n{'=' * 70}")
    print(f"RESULTS  ({n_trials} realisations,  mode = {update_mode})")
    print(f"{'=' * 70}\n")

    hdr = f"{'Metric':<25} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, arr, fmt=".2f"):
        print(f"{label:<25} | {np.mean(arr):>10{fmt}} | {np.std(arr):>10{fmt}} | "
              f"{np.min(arr):>10{fmt}} | {np.max(arr):>10{fmt}}")

    row("EE (Mbit/J)",      ee_M)
    row("Power (mW)",       pw_mW,  ".3f")
    row("SINR (linear)",    sinr_a)
    sinr_dB = 10 * np.log10(np.maximum(sinr_a, 1e-20))
    row("SINR (dB)",        sinr_dB)
    row("Rate (bit/s/Hz)",  rate_a, ".3f")

    print(f"\nConvergence : {np.sum(all_converged)}/{n_trials} trials")
    print(f"QoS met     : {np.sum(all_qos_met)}/{n_trials}  "
          f"({100*np.mean(all_qos_met):.1f} %)")
    print(f"Avg iters   : {np.mean(all_iters):.1f}  (max {np.max(all_iters)})")
    print(f"Wall-clock  : {elapsed:.1f} s  ({elapsed/n_trials*1e3:.1f} ms / trial)")

    # ---- save ----
    results = {
        "algorithm": "Zappone et al. — Decentralised EE (BRD + Dinkelbach)",
        "update_mode": update_mode,
        "n_trials":    n_trials,
        "config": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in cfg.items()},
        "statistics": {
            "ee_mbit_j": {
                "mean":   float(np.mean(ee_M)),
                "std":    float(np.std(ee_M)),
                "min":    float(np.min(ee_M)),
                "max":    float(np.max(ee_M)),
                "median": float(np.median(ee_M)),
                "p5":     float(np.percentile(ee_M, 5)),
                "p95":    float(np.percentile(ee_M, 95)),
            },
            "power_mw": {
                "mean": float(np.mean(pw_mW)),
                "std":  float(np.std(pw_mW)),
            },
            "rate_bps_hz": {
                "mean": float(np.mean(rate_a)),
                "std":  float(np.std(rate_a)),
            },
            "convergence": {
                "converged_fraction": float(np.mean(all_converged)),
                "avg_iterations":    float(np.mean(all_iters)),
                "max_iterations":    int(np.max(all_iters)),
            },
            "qos": {
                "satisfaction_rate": float(np.mean(all_qos_met)),
            },
        },
        "per_trial_ee_mbit_j":  ee_M.tolist(),
        "per_trial_power_mw":   pw_mW.tolist(),
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/article_algorithm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ---- plots ----
    _plot_results(ee_M, pw_mW, all_iters, cfg)

    return results


# ======================================================================
# Plotting
# ======================================================================

def _plot_results(ee_M, pw_mW, iters, cfg):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping plots)")
        return

    fig_dir = os.path.join("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. EE distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(ee_M, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(np.mean(ee_M), color="red", ls="--", label=f"mean = {np.mean(ee_M):.1f}")
    axes[0].set_xlabel("Energy Efficiency (Mbit/J)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Article Algorithm — EE Distribution")
    axes[0].legend()

    # 2. Power distribution
    axes[1].hist(pw_mW, bins=30, edgecolor="black", alpha=0.7, color="darkorange")
    axes[1].axvline(np.mean(pw_mW), color="red", ls="--", label=f"mean = {np.mean(pw_mW):.3f}")
    axes[1].set_xlabel("Transmit Power (mW)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Article Algorithm — Power Distribution")
    axes[1].legend()

    # 3. Convergence iterations
    axes[2].hist(iters, bins=range(1, max(iters) + 2), edgecolor="black",
                 alpha=0.7, color="seagreen")
    axes[2].set_xlabel("Outer Iterations to Converge")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Convergence Speed")

    plt.tight_layout()
    path = os.path.join(fig_dir, "article_algorithm_evaluation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the article's decentralised EE algorithm"
    )
    parser.add_argument("--trials",  type=int, default=200,
                        help="Number of channel realisations")
    parser.add_argument("--mode",    type=str, default="sequential",
                        choices=["sequential", "synchronous"],
                        help="Power update mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration details for each trial")
    args = parser.parse_args()

    run_evaluation(
        n_trials=args.trials,
        update_mode=args.mode,
        verbose_trials=args.verbose,
    )
