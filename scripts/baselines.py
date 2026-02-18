"""
Baselines for comparison:
  1. Nash Equilibrium via Dinkelbach (Algorithm 2 from the article)
  2. Max Power (all users at p_max)
  3. Random Power
  4. Equal Power at various levels

These baselines provide EE targets the DQN should match or exceed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import MassiveMIMOChannel


def dinkelbach_best_response(channel: MassiveMIMOChannel, powers: np.ndarray,
                              user_idx: int, p_circuit: float, p_max: float,
                              p_min: float = 1e-6, n_points: int = 200) -> float:
    """
    Find the EE-maximizing power for user k via golden section search.
    
    Solves: max_{p_k in [p_min, p_max]} B * log2(1 + SINR(p_k)) / (p_c + p_k)
    
    where SINR(p_k) = alpha_k * p_k / (sigma^2 + phi_k * p_k + I_k)
    and I_k = sum_{j!=k} omega_kj * p_j (treated as constant).
    """
    B = channel.bandwidth
    alpha_k, phi_k, omega_kj, sigma_sq = channel.compute_sinr_coefficients(user_idx)
    
    # Interference from other users (fixed)
    I_k = sum(omega_kj[j] * powers[j] for j in range(channel.n_users) if j != user_idx)
    s = sigma_sq + I_k  # effective noise + interference
    
    def ee_func(p):
        sinr = alpha_k * p / max(s + phi_k * p, 1e-30)
        rate = np.log2(1 + sinr)
        return B * rate / (p_circuit + p)
    
    # Golden section search on log scale 
    log_pmin = np.log10(max(p_min, 1e-10))
    log_pmax = np.log10(p_max)
    
    # Evaluate on grid first to find region
    log_ps = np.linspace(log_pmin, log_pmax, n_points)
    ees = [ee_func(10**lp) for lp in log_ps]
    best_idx = np.argmax(ees)
    
    # Refine around best
    if best_idx > 0 and best_idx < n_points - 1:
        log_lo = log_ps[best_idx - 1]
        log_hi = log_ps[best_idx + 1]
        log_ps2 = np.linspace(log_lo, log_hi, 100)
        ees2 = [ee_func(10**lp) for lp in log_ps2]
        best_idx2 = np.argmax(ees2)
        return 10 ** log_ps2[best_idx2]
    
    return 10 ** log_ps[best_idx]


def nash_equilibrium_brd(channel: MassiveMIMOChannel, p_circuit: float,
                          p_max: float, p_min: float = 1e-6,
                          n_outer: int = 50, tol: float = 1e-4):
    """
    Find the Nash Equilibrium using Best Response Dynamics (Algorithm 2).
    """
    K = channel.n_users
    powers = np.ones(K) * p_max * 0.01
    
    for iteration in range(n_outer):
        old_powers = powers.copy()
        
        for k in range(K):
            powers[k] = dinkelbach_best_response(
                channel, powers, k, p_circuit, p_max, p_min
            )
        
        # Check convergence
        if np.max(np.abs(powers - old_powers)) < tol * p_max:
            break
    
    return powers, iteration + 1


def evaluate_baselines(n_trials: int = 200):
    """Evaluate all baselines over multiple channel realizations."""
    channel = MassiveMIMOChannel("config/network.yaml")
    K = channel.n_users
    B = channel.bandwidth
    
    # Power constraints
    import yaml
    with open("config/network.yaml") as f:
        cfg = yaml.safe_load(f)['network']
    p_max_dBm = cfg['power']['max_power_dBm']
    p_max = 10 ** (p_max_dBm / 10) * 1e-3
    p_circuit_dBm = cfg['power']['circuit_power_dBm']
    p_circuit = 10 ** (p_circuit_dBm / 10) * 1e-3
    p_min_dBm = cfg['power']['min_power_dBm']
    p_min = 10 ** (p_min_dBm / 10) * 1e-3
    
    print(f"K={K}, M={channel.n_antennas}, B={B/1e6:.0f} MHz")
    print(f"p_min={p_min*1000:.3f} mW, p_max={p_max*1000:.1f} mW, p_c={p_circuit*1000:.1f} mW")
    print(f"gamma_max={1/channel.epsilon_sq:.0f} ({10*np.log10(1/channel.epsilon_sq):.1f} dB)")
    print(f"\nRunning {n_trials} trials...\n")
    
    results = {
        'Nash (Dinkelbach BRD)': [],
        'Optimal Uniform Power': [],
        'Max Power': [],
        'Random Power': [],
        'Min Power': [],
    }
    
    nash_powers_avg = []
    nash_iters = []
    
    for trial in range(n_trials):
        channel.reset()
        
        # 1. Nash Equilibrium (BRD)
        ne_powers, n_iter = nash_equilibrium_brd(channel, p_circuit, p_max, p_min)
        sinrs_ne = channel.compute_all_sinr(ne_powers)
        ees_ne = [channel.compute_ee(channel.compute_rate(s), ne_powers[k], p_circuit)
                  for k, s in enumerate(sinrs_ne)]
        results['Nash (Dinkelbach BRD)'].append(np.mean(ees_ne))
        nash_powers_avg.append(np.mean(ne_powers))
        nash_iters.append(n_iter)
        
        # 2. Find optimal uniform power (brute force)
        best_ee_uniform = 0
        for p_test in np.geomspace(p_min, p_max, 50):
            powers = np.ones(K) * p_test
            sinrs = channel.compute_all_sinr(powers)
            ees = [channel.compute_ee(channel.compute_rate(s), p_test, p_circuit) for s in sinrs]
            avg_ee = np.mean(ees)
            if avg_ee > best_ee_uniform:
                best_ee_uniform = avg_ee
        results['Optimal Uniform Power'].append(best_ee_uniform)
        
        # 3. Max Power
        powers = np.ones(K) * p_max
        sinrs = channel.compute_all_sinr(powers)
        ees = [channel.compute_ee(channel.compute_rate(s), p_max, p_circuit) for s in sinrs]
        results['Max Power'].append(np.mean(ees))
        
        # 4. Random Power
        powers = np.random.uniform(p_min, p_max, K)
        sinrs = channel.compute_all_sinr(powers)
        ees = [channel.compute_ee(channel.compute_rate(s), powers[k], p_circuit)
               for k, s in enumerate(sinrs)]
        results['Random Power'].append(np.mean(ees))
        
        # 5. Min Power
        powers = np.ones(K) * p_min
        sinrs = channel.compute_all_sinr(powers)
        ees = [channel.compute_ee(channel.compute_rate(s), p_min, p_circuit) for s in sinrs]
        results['Min Power'].append(np.mean(ees))
    
    # Print results
    print(f"{'Baseline':<25} | {'Mean EE':>12} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 70)
    for name, ees in results.items():
        ees_mbit = np.array(ees) / 1e6
        print(f"{name:<25} | {np.mean(ees_mbit):>9.1f} M  | {np.std(ees_mbit):>6.1f}  | "
              f"{np.min(ees_mbit):>6.1f}  | {np.max(ees_mbit):>6.1f}")
    
    print(f"\n  Nash avg power: {np.mean(nash_powers_avg)*1000:.3f} mW "
          f"({10*np.log10(max(np.mean(nash_powers_avg)*1000, 1e-20)):.1f} dBm)")
    print(f"  Nash avg iterations: {np.mean(nash_iters):.1f}")
    print(f"\n  Article target (decentralized): 250-310 Mbit/J")
    
    return results


if __name__ == "__main__":
    results = evaluate_baselines(n_trials=100)
