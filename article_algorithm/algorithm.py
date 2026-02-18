"""
Decentralized Power Control via Best Response Dynamics
======================================================
Outer loop of Algorithm 2 from Zappone et al. (2015/2021).

Each user independently maximises its energy efficiency using
Dinkelbach's method (inner loop), treating other users' powers as
fixed.  The algorithm converges to the unique Generalised Nash
Equilibrium (GNE) if the feasibility conditions are satisfied.

Two update modes:
  - sequential (default, Gauss-Seidel):  user k sees updated powers
        of users 1..k−1 within the same iteration  →  faster convergence.
  - synchronous (Jacobi):  all users update based on the *previous*
        iteration's powers  →  matches the theoretical convergence proof.
"""

import numpy as np
import sys
import os

# Allow importing from the project's src/ directory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "src"))

from core.network import MassiveMIMOChannel
from article_algorithm.dinkelbach import (
    dinkelbach_solve,
    compute_min_qos_power,
)


class DecentralizedEEAlgorithm:
    """
    Implements Algorithm 2 from Zappone et al. for a single resource block.

    Per-iteration steps (for every user k):
      (a) Measure current SINR γ_k
      (b) Compute equivalent channel gain μ_k via Eq. (14)
      (c) Compute minimum QoS power  p̲_k
      (d) Solve Dinkelbach for unconstrained optimal power  p*_k
      (e) Project:  p_k ← min( p̄, max( p*_k, p̲_k ) )
    """

    def __init__(self,
                 channel: MassiveMIMOChannel,
                 p_max: float,
                 p_circuit: float,
                 qos_target_rate: float = None,
                 gamma_min: float = None,
                 outer_tol: float = 1e-4,
                 inner_tol: float = 1e-6,
                 max_outer_iter: int = 100,
                 max_inner_iter: int = 100,
                 update_mode: str = "sequential"):
        """
        Args:
            channel:          MassiveMIMOChannel (already reset)
            p_max:            Maximum transmit power per user  (W)
            p_circuit:        Circuit power consumption  (W)
            qos_target_rate:  Target rate θ_k  (bit/s/Hz) — optional
            gamma_min:        Minimum SINR target  (alternative to rate)
            outer_tol:        Convergence tolerance (relative to p_max)
            inner_tol:        Dinkelbach tolerance
            max_outer_iter:   Max best-response iterations
            max_inner_iter:   Max Dinkelbach iterations per user
            update_mode:      "sequential" (Gauss-Seidel) or "synchronous" (Jacobi)
        """
        self.channel = channel
        self.K = channel.n_users
        self.B = channel.bandwidth
        self.p_max = p_max
        self.p_circuit = p_circuit
        self.outer_tol = outer_tol
        self.inner_tol = inner_tol
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.update_mode = update_mode

        # γ̄ = α_k / φ_k = 1 / ε²  (same for all users in massive MIMO)
        self.gamma_max = channel.compute_max_sinr()

        # QoS constraint:  γ̲ = 2^θ − 1
        if gamma_min is not None:
            self.gamma_min = gamma_min
        elif qos_target_rate is not None:
            self.gamma_min = 2.0 ** qos_target_rate - 1.0
        else:
            self.gamma_min = 0.0

        self.max_rate = np.log2(1.0 + self.gamma_max)

    # ------------------------------------------------------------------
    # μ_k from observed SINR  (Equation 14)
    # ------------------------------------------------------------------
    def compute_mu_from_sinr(self, sinr_k: float, p_k: float) -> float:
        """
        Equivalent channel gain from observed SINR:
          μ_k = (γ_k / p_k) · (1 − γ_k / γ̄_k)⁻¹

        This is the *decentralised* computation: user k only needs
        its own transmit power, the measured SINR, and γ̄.
        """
        if p_k <= 0 or sinr_k <= 0:
            return 1e-10
        ratio = sinr_k / self.gamma_max
        if ratio >= 1.0 - 1e-12:
            ratio = 1.0 - 1e-12
        return (sinr_k / p_k) / (1.0 - ratio)

    # ------------------------------------------------------------------
    # Main solver
    # ------------------------------------------------------------------
    def solve(self, init_powers: np.ndarray = None, verbose: bool = False):
        """
        Run the decentralised best-response dynamics.

        Args:
            init_powers: Initial power vector (K,).  Default: 1 % of p_max.
            verbose:     Print per-iteration details.

        Returns:
            powers   : Final power allocation (K,) in Watts
            ee_values: Final per-user EE (K,) in bit/J
            history  : Dict with convergence trace
        """
        # ----- initialisation -----
        if init_powers is not None:
            powers = np.clip(init_powers.copy(), 1e-10, self.p_max)
        else:
            powers = np.ones(self.K) * self.p_max * 0.01

        history = {
            "powers":       [powers.copy()],
            "ee":           [],
            "sinr":         [],
            "rates":        [],
            "convergence":  [],
            "inner_iters":  [],
        }

        for t in range(self.max_outer_iter):
            old_powers = powers.copy()
            inner_iters_round = []

            if self.update_mode == "synchronous":
                # Jacobi: compute all new powers from old_powers
                new_powers = powers.copy()

            for k in range(self.K):
                # (a) Measure SINR (uses current powers for sequential,
                #     old_powers for synchronous)
                ref_powers = old_powers if self.update_mode == "synchronous" else powers
                sinr_k = self.channel.compute_sinr(ref_powers, k)

                # (b) Equivalent channel gain  μ_k  (Eq. 14)
                mu_k = self.compute_mu_from_sinr(sinr_k, ref_powers[k])
                mu_k = max(mu_k, 1e-10)

                # (c) Minimum power for QoS
                if self.gamma_min > 0:
                    p_qos_min = compute_min_qos_power(
                        mu_k, self.gamma_max, self.gamma_min
                    )
                    if np.isinf(p_qos_min):
                        p_qos_min = self.p_max   # infeasible → saturate
                else:
                    p_qos_min = 0.0

                # (d) Dinkelbach solve for unconstrained optimal
                _, p_star, _, n_inner = dinkelbach_solve(
                    mu_k, self.gamma_max, self.p_circuit, self.B,
                    tol=self.inner_tol, max_iter=self.max_inner_iter,
                )
                inner_iters_round.append(n_inner)

                # (e) Project onto feasible set  [p̲, p̄]
                p_new = min(self.p_max, max(p_star, p_qos_min))
                p_new = max(p_new, 1e-10)

                if self.update_mode == "synchronous":
                    new_powers[k] = p_new
                else:
                    powers[k] = p_new

            if self.update_mode == "synchronous":
                powers = new_powers

            # ----- record metrics -----
            sinrs = self.channel.compute_all_sinr(powers)
            rates = np.array([self.channel.compute_rate(s) for s in sinrs])
            ees   = np.array([
                self.channel.compute_ee(rates[k], powers[k], self.p_circuit)
                for k in range(self.K)
            ])

            max_change = np.max(np.abs(powers - old_powers))

            history["powers"].append(powers.copy())
            history["ee"].append(ees.copy())
            history["sinr"].append(sinrs.copy())
            history["rates"].append(rates.copy())
            history["convergence"].append(max_change)
            history["inner_iters"].append(inner_iters_round)

            if verbose:
                print(
                    f"  Iter {t+1:3d}:  max_Δp = {max_change:.2e},  "
                    f"mean_EE = {np.mean(ees)/1e6:.2f} Mbit/J,  "
                    f"powers(mW) = [{', '.join(f'{p*1e3:.3f}' for p in powers)}]"
                )

            # ----- convergence check -----
            if max_change < self.outer_tol * self.p_max:
                if verbose:
                    print(f"  ✓ Converged in {t+1} iterations.")
                break

        # ----- final evaluation -----
        sinrs = self.channel.compute_all_sinr(powers)
        rates = np.array([self.channel.compute_rate(s) for s in sinrs])
        ees   = np.array([
            self.channel.compute_ee(rates[k], powers[k], self.p_circuit)
            for k in range(self.K)
        ])

        history["n_outer_iter"] = t + 1
        history["converged"]    = (max_change < self.outer_tol * self.p_max)

        return powers, ees, history


# ======================================================================
# Convenience wrapper
# ======================================================================

def run_single_realization(channel: MassiveMIMOChannel,
                           p_max: float,
                           p_circuit: float,
                           qos_target_rate: float = None,
                           update_mode: str = "sequential",
                           verbose: bool = False):
    """
    Run the algorithm for one channel realisation.
    The channel must have been reset() before calling this.
    """
    algo = DecentralizedEEAlgorithm(
        channel=channel,
        p_max=p_max,
        p_circuit=p_circuit,
        qos_target_rate=qos_target_rate,
        update_mode=update_mode,
    )
    return algo.solve(verbose=verbose)
