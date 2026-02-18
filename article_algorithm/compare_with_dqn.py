"""
Compare Article Algorithm vs DQN vs Baselines
==============================================
Runs all methods on the **same** channel realisations so that
the comparison is fair (paired samples).

Usage:
    python article_algorithm/compare_with_dqn.py
    python article_algorithm/compare_with_dqn.py --trials 300

Outputs:
    results/comparison_results.json
    results/figures/comparison_*.png
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
# Configuration
# ======================================================================

def load_config():
    with open("config/network.yaml") as f:
        cfg = yaml.safe_load(f)["network"]
    with open("config/dqn.yaml") as f:
        dqn_cfg = yaml.safe_load(f)["dqn"]

    p_max     = 10 ** (cfg["power"]["max_power_dBm"]     / 10) * 1e-3
    p_min     = 10 ** (cfg["power"]["min_power_dBm"]     / 10) * 1e-3
    p_circuit = 10 ** (cfg["power"]["circuit_power_dBm"] / 10) * 1e-3

    eps_sq    = (cfg["channel"]["error_magnitude_bs"] ** 2
                 + cfg["channel"]["error_magnitude_ue"] ** 2)
    gamma_max = 1.0 / eps_sq
    max_rate  = np.log2(1.0 + gamma_max)

    qos_frac  = cfg["qos"].get("target_rate_fraction",
                                cfg["qos"]["min_rate_percentage"] / 100.0)
    qos_target_rate = qos_frac * max_rate

    n_actions = dqn_cfg["actions"]["n_levels"]
    power_levels = np.geomspace(p_min, p_max, n_actions)

    return {
        "p_max":            p_max,
        "p_min":            p_min,
        "p_circuit":        p_circuit,
        "gamma_max":        gamma_max,
        "max_rate":         max_rate,
        "qos_target_rate":  qos_target_rate,
        "gamma_min":        2.0 ** qos_target_rate - 1.0,
        "n_actions":        n_actions,
        "power_levels":     power_levels,
        "K":                cfg["n_users"],
    }


# ======================================================================
# DQN evaluation helpers
# ======================================================================

def load_dqn_agents(K, state_dim, action_dim, device="cpu"):
    """Load trained DQN agents (one per user)."""
    try:
        from core.agent import DQNAgent
    except ImportError:
        print("  ⚠ Could not import DQNAgent — skipping DQN comparison.")
        return None

    agents = []
    for k in range(K):
        path = f"models/dqn_agent_user{k}_final.pt"
        if not os.path.exists(path):
            print(f"  ⚠ Model file not found: {path}")
            return None
        agent = DQNAgent(user_id=k,
                         state_dim=state_dim,
                         action_dim=action_dim,
                         config_path="config/dqn.yaml",
                         device=device)
        agent.load(path)
        agent.set_epsilon(0.0)   # greedy
        agents.append(agent)
    return agents


def evaluate_dqn_on_channel(agents, channel, cfg, n_steps=50):
    """
    Run DQN agents greedily on a *fixed* channel realisation for
    n_steps (to let actions stabilise), then return the final powers.
    """
    from core.environment import PowerControlEnv

    K = cfg["K"]
    env = PowerControlEnv.__new__(PowerControlEnv)
    # Manually wire the env to use the same channel object
    env.channel = channel
    env.n_users = K
    env.single_user_mode = False
    env.user_idx = 0
    env.p_min = cfg["p_min"]
    env.p_max = cfg["p_max"]
    env.p_circuit = cfg["p_circuit"]
    env.power_levels = cfg["power_levels"]
    env.n_actions = cfg["n_actions"]
    env.state_dim = 6
    env.gamma_max = cfg["gamma_max"]
    env.max_rate  = cfg["max_rate"]
    env.theta_k   = cfg["qos_target_rate"]
    env.qos_enabled = True
    env.min_rate_pct = 20.0
    env.step_count = 0
    env.max_steps = n_steps
    env.current_powers = np.random.uniform(cfg["p_min"],
                                           cfg["p_max"] * 0.1,
                                           size=K)

    with open("config/dqn.yaml") as f:
        dqn_cfg = yaml.safe_load(f)["dqn"]
    env.reward_cfg = dqn_cfg["reward"]
    env.action_cfg = dqn_cfg["actions"]
    env.ee_ref = dqn_cfg["reward"].get("ee_reference", 1e8)
    env.net_cfg = yaml.safe_load(open("config/network.yaml"))["network"]

    # Get initial states
    states = {}
    for k in range(K):
        states[k] = env._get_state(k)

    # Run greedy for n_steps
    for _ in range(n_steps):
        actions = {}
        for k in range(K):
            actions[k] = agents[k].select_action(states[k], training=False)
        states, _, done, _ = env.step(actions)
        if done:
            break

    return env.current_powers.copy()


# ======================================================================
# Baselines
# ======================================================================

def optimal_uniform_power(channel, cfg, n_grid=100):
    """Brute-force search over 100 uniform power levels."""
    K = cfg["K"]
    best_ee, best_p = 0.0, cfg["p_min"]
    for p in np.geomspace(cfg["p_min"], cfg["p_max"], n_grid):
        powers = np.ones(K) * p
        sinrs  = channel.compute_all_sinr(powers)
        ees    = [channel.compute_ee(channel.compute_rate(s), p, cfg["p_circuit"])
                  for s in sinrs]
        avg = np.mean(ees)
        if avg > best_ee:
            best_ee, best_p = avg, p
    return np.ones(K) * best_p


def max_power(channel, cfg):
    return np.ones(cfg["K"]) * cfg["p_max"]


def random_power(channel, cfg):
    return np.random.uniform(cfg["p_min"], cfg["p_max"], cfg["K"])


# ======================================================================
# Main comparison
# ======================================================================

def run_comparison(n_trials: int = 200, device: str = "cpu"):
    channel = MassiveMIMOChannel("config/network.yaml")
    cfg     = load_config()
    K       = cfg["K"]

    print("=" * 70)
    print("COMPARISON : Article Algorithm  vs  DQN  vs  Baselines")
    print("=" * 70)
    print(f"K = {K},  M = {channel.n_antennas},  B = {channel.bandwidth/1e6:.0f} MHz")
    print(f"Trials = {n_trials}\n")

    # Try loading DQN agents
    agents = load_dqn_agents(K, state_dim=6, action_dim=cfg["n_actions"],
                             device=device)
    include_dqn = agents is not None

    methods = ["Article (BRD+Dinkelbach)", "Optimal Uniform", "Max Power", "Random Power"]
    if include_dqn:
        methods.insert(1, "DQN (trained)")

    results = {m: [] for m in methods}
    powers_results = {m: [] for m in methods}

    t0 = time.time()

    for trial in range(n_trials):
        channel.reset()

        # 1. Article algorithm
        algo = DecentralizedEEAlgorithm(
            channel=channel,
            p_max=cfg["p_max"],
            p_circuit=cfg["p_circuit"],
            qos_target_rate=cfg["qos_target_rate"],
        )
        a_powers, a_ees, _ = algo.solve()
        results["Article (BRD+Dinkelbach)"].append(float(np.mean(a_ees)))
        powers_results["Article (BRD+Dinkelbach)"].append(float(np.mean(a_powers)))

        # 2. DQN (if available)
        if include_dqn:
            dqn_powers = evaluate_dqn_on_channel(agents, channel, cfg)
            sinrs = channel.compute_all_sinr(dqn_powers)
            ees   = [channel.compute_ee(channel.compute_rate(s), dqn_powers[k],
                                        cfg["p_circuit"])
                     for k, s in enumerate(sinrs)]
            results["DQN (trained)"].append(float(np.mean(ees)))
            powers_results["DQN (trained)"].append(float(np.mean(dqn_powers)))

        # 3. Optimal uniform
        u_powers = optimal_uniform_power(channel, cfg)
        sinrs = channel.compute_all_sinr(u_powers)
        ees   = [channel.compute_ee(channel.compute_rate(s), u_powers[k],
                                    cfg["p_circuit"])
                 for k, s in enumerate(sinrs)]
        results["Optimal Uniform"].append(float(np.mean(ees)))
        powers_results["Optimal Uniform"].append(float(np.mean(u_powers)))

        # 4. Max power
        m_powers = max_power(channel, cfg)
        sinrs = channel.compute_all_sinr(m_powers)
        ees   = [channel.compute_ee(channel.compute_rate(s), m_powers[k],
                                    cfg["p_circuit"])
                 for k, s in enumerate(sinrs)]
        results["Max Power"].append(float(np.mean(ees)))
        powers_results["Max Power"].append(float(np.mean(m_powers)))

        # 5. Random power
        r_powers = random_power(channel, cfg)
        sinrs = channel.compute_all_sinr(r_powers)
        ees   = [channel.compute_ee(channel.compute_rate(s), r_powers[k],
                                    cfg["p_circuit"])
                 for k, s in enumerate(sinrs)]
        results["Random Power"].append(float(np.mean(ees)))
        powers_results["Random Power"].append(float(np.mean(r_powers)))

        if (trial + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{trial+1:4d}/{n_trials}]  elapsed {elapsed:.1f}s")

    elapsed = time.time() - t0

    # ---- summary table ----
    print(f"\n{'=' * 75}")
    print(f"{'Method':<30} | {'Mean EE':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print(f"{'':<30} | {'(Mbit/J)':>10} | {'':>8} | {'':>8} | {'':>8}")
    print("-" * 75)
    for m in methods:
        arr = np.array(results[m]) / 1e6
        print(f"{m:<30} | {np.mean(arr):>10.1f} | {np.std(arr):>8.1f} | "
              f"{np.min(arr):>8.1f} | {np.max(arr):>8.1f}")

    print(f"\n{'Method':<30} | {'Mean Power (mW)':>15}")
    print("-" * 50)
    for m in methods:
        arr = np.array(powers_results[m]) * 1e3
        print(f"{m:<30} | {np.mean(arr):>15.3f}")

    print(f"\nTotal time: {elapsed:.1f}s")

    # ---- percentage comparison ----
    article_mean = np.mean(results["Article (BRD+Dinkelbach)"]) / 1e6
    print(f"\n--- Relative to Article Algorithm ({article_mean:.1f} Mbit/J) ---")
    for m in methods:
        if m == "Article (BRD+Dinkelbach)":
            continue
        m_mean = np.mean(results[m]) / 1e6
        diff_pct = (m_mean - article_mean) / article_mean * 100
        print(f"  {m:<30}  {m_mean:>8.1f} Mbit/J  ({diff_pct:+.1f} %)")

    # ---- save ----
    output = {
        "n_trials": n_trials,
        "methods": methods,
        "ee_per_trial": {m: (np.array(v) / 1e6).tolist() for m, v in results.items()},
        "summary": {
            m: {
                "mean_ee_mbit_j": float(np.mean(np.array(results[m]) / 1e6)),
                "std_ee_mbit_j":  float(np.std(np.array(results[m]) / 1e6)),
                "mean_power_mw":  float(np.mean(np.array(powers_results[m]) * 1e3)),
            }
            for m in methods
        },
    }
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved  results/comparison_results.json")

    # ---- plots ----
    _plot_comparison(results, methods)

    return output


# ======================================================================
# Plotting
# ======================================================================

def _plot_comparison(results, methods):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping plots)")
        return

    fig_dir = os.path.join("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    colours = {
        "Article (BRD+Dinkelbach)": "steelblue",
        "DQN (trained)":           "crimson",
        "Optimal Uniform":         "darkorange",
        "Max Power":               "grey",
        "Random Power":            "mediumpurple",
    }

    # --- 1. Box plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [np.array(results[m]) / 1e6 for m in methods]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, notch=True)
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(colours.get(m, "lightgrey"))
    ax.set_ylabel("Energy Efficiency (Mbit/J)")
    ax.set_title("Comparison of Power Control Methods")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(fig_dir, "comparison_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")

    # --- 2. CDF ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        arr = np.sort(np.array(results[m]) / 1e6)
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf, label=m, color=colours.get(m, "black"), linewidth=2)
    ax.set_xlabel("Energy Efficiency (Mbit/J)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Energy Efficiency")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "comparison_cdf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")

    # --- 3. Bar chart (mean ± std) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(np.array(results[m]) / 1e6) for m in methods]
    stds  = [np.std(np.array(results[m]) / 1e6) for m in methods]
    bars  = ax.bar(methods, means, yerr=stds, capsize=5,
                   color=[colours.get(m, "lightgrey") for m in methods],
                   edgecolor="black", alpha=0.85)
    ax.set_ylabel("Energy Efficiency (Mbit/J)")
    ax.set_title("Mean EE ± Std")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mean:.0f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Article Algorithm vs DQN vs Baselines"
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_comparison(n_trials=args.trials, device=args.device)
