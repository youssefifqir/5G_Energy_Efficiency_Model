"""
Numerical Validation — Verify EE matches article (~250-310 Mbit/J).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import MassiveMIMOChannel
from core.environment import PowerControlEnv

def validate():
    ch = MassiveMIMOChannel("config/network.yaml")
    K, M, B = ch.n_users, ch.n_antennas, ch.bandwidth
    p_c = 0.01  # 10 mW

    print(f"K={K}, M={M}, B={B/1e6:.0f} MHz")
    print(f"eps^2={ch.epsilon_sq:.4f} => gamma_max={1/ch.epsilon_sq:.0f} ({10*np.log10(1/ch.epsilon_sq):.1f} dB)")
    print(f"noise={ch.noise_power:.2e} W, p_c={p_c*1000:.0f} mW")

    # Test power levels (logarithmically spaced from 0.001 mW to 200 mW)
    test_powers = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0]  # mW
    n_trials = 200

    print(f"\nAveraging over {n_trials} random channel realizations...")
    print(f"\n{'P(mW)':>8} | {'P(dBm)':>8} | {'SINR(dB)':>9} | {'Rate':>8} | {'EE(Mbit/J)':>12} | {'std':>6}")
    print("-" * 63)

    for p_mW in test_powers:
        p_W = p_mW / 1000.0
        ee_list, sinr_list, rate_list = [], [], []

        for _ in range(n_trials):
            ch.reset()
            powers = np.ones(K) * p_W
            sinrs = ch.compute_all_sinr(powers)
            rates = [ch.compute_rate(s) for s in sinrs]
            ees = [ch.compute_ee(r, p_W, p_c) for r in rates]
            ee_list.append(np.mean(ees))
            sinr_list.append(np.mean(10 * np.log10(np.maximum(sinrs, 1e-20))))
            rate_list.append(np.mean(rates))

        avg_ee = np.mean(ee_list) / 1e6
        std_ee = np.std(ee_list) / 1e6
        avg_sinr = np.mean(sinr_list)
        avg_rate = np.mean(rate_list)
        p_dBm = 10 * np.log10(p_mW)

        marker = " <-- OPTIMAL" if avg_ee == max(np.mean(e)/1e6 for e in [ee_list]) else ""
        print(f"{p_mW:8.3f} | {p_dBm:8.1f} | {avg_sinr:9.1f} | {avg_rate:8.3f} | {avg_ee:12.1f} | {std_ee:6.1f}{marker}")

    # Find actual optimal
    best_p, best_ee = 0, 0
    for p_mW in np.geomspace(0.001, 200, 100):
        p_W = p_mW / 1000
        ees = []
        for _ in range(50):
            ch.reset()
            powers = np.ones(K) * p_W
            sinrs = ch.compute_all_sinr(powers)
            rates = [ch.compute_rate(s) for s in sinrs]
            avg_ee = np.mean([ch.compute_ee(r, p_W, p_c) for r in rates])
            ees.append(avg_ee)
        mean_ee = np.mean(ees)
        if mean_ee > best_ee:
            best_ee = mean_ee
            best_p = p_mW

    print(f"\n  OPTIMAL: p={best_p:.3f} mW ({10*np.log10(best_p):.1f} dBm) => EE={best_ee/1e6:.1f} Mbit/J")
    print(f"  Article target: ~250-310 Mbit/J (decentralized, K=5, M=50)")

    # Quick env test
    print("\n--- Environment Test ---")
    env = PowerControlEnv("config/network.yaml", "config/dqn.yaml", user_idx=0)
    state = env.reset()
    print(f"  State dim: {env.get_state_dim()}, Action dim: {env.get_action_dim()}")
    print(f"  Power range: {env.power_levels[0]*1000:.3f} - {env.power_levels[-1]*1000:.1f} mW")
    print(f"  gamma_max: {env.gamma_max:.0f} ({10*np.log10(env.gamma_max):.1f} dB)")
    print(f"  theta_k (QoS): {env.theta_k:.2f} bit/s/Hz")
    print(f"  Initial state: {state}")

    # Try a few actions
    for a in [0, 8, 16, 24, 31]:
        state, reward, done, info = env.step(a)
        print(f"  Action {a:2d}: p={info['power_dBm']:.1f}dBm, SINR={info['sinr_dB']:.1f}dB, "
              f"EE={info['ee_Mbit_J']:.1f}Mbit/J, R={reward:.3f}")

if __name__ == "__main__":
    validate()
